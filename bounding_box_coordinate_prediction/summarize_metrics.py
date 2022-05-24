import os
import json
import argparse
import pathlib
import numpy as np

from utilities import compute_l2_loss
from utilities import get_datetime
from utilities import JsonNumericEncoder, JsonNumericDecoder


max_len_model_id = 37

def load_predictions(directory, filenames, verbose=True):
    """
    Returns:
        predictions: Dict of form {dataset_id: {"models": {model_id: prediction}}}, where
            prediction is a dict of form {model_id: str, dataset_id: str, 
            "images": {image_id: {"size": [], "iou": float, "target": [], "prediction": [], "feature": []}}}
    """
    predictions = dict() # {dataset_id: {"models": {model_id: prediction}}}
    dataset_ids = []
    model_ids = []
    union_image_ids = dict() # {dataset_id: [image_id, ...], ...}

    # Iterate over all prediction files
    for i, filename in enumerate(filenames):
        print("{} Loading from {}...".format(get_datetime(), filename))
        # Load prediction
        path = os.path.join(directory, filename)
        with open(path, 'r') as f:
            # {model_id: str, dataset_id: str, "images": {image_id: {"size": [], "iou": float, "target": [], "prediction": [], "feature": []}}}
            prediction = json.load(f) #, object_pairs_hook=JsonNumericDecoder)

        dataset_id = prediction["dataset_id"]
        model_id = prediction["model_id"]
        if dataset_id not in predictions:
            predictions[dataset_id] = dict()
            predictions[dataset_id]["models"] = dict()
        predictions[dataset_id]["models"][model_id] = prediction
        if verbose:
            print("{} {} {} images {}, predictions {}".format(
                get_datetime(),
                prediction["dataset_id"],
                prediction["model_id"],
                len(prediction["images"]),
                sum([len(prediction["images"][i]["prediction"]) for i in prediction["images"]]),
            ))

        # Record all dataset ids and model ids
        if dataset_id not in dataset_ids:
            dataset_ids.append(dataset_id)
        if model_id not in model_ids:
            model_ids.append(model_id)

        # Record all image ids
        if dataset_id not in union_image_ids:
            union_image_ids[dataset_id] = predictions[dataset_id]["models"][model_id]["images"].keys()
        else:
            union_image_ids[dataset_id] = union_image_ids[dataset_id] | predictions[dataset_id]["models"][model_id]["images"].keys()

    for dataset_id in dataset_ids:
        union_image_ids[dataset_id] = sorted(union_image_ids[dataset_id])

    return predictions, dataset_ids, model_ids, union_image_ids


def summarize_metrics(metrics, dataset_ids, model_ids, union_image_ids, iou_threshold, loss_on_common_tps=True, normalize_coordinates=True):
    """
    """
    common_tps = dict()
    predictions = dict()
    losses = dict()
    features = dict()

    # Iterate over all datasets
    for dataset_id in dataset_ids:
        print("{} Summarizing {}...".format(get_datetime(), dataset_id))
        # Load prediction
        common_tps[dataset_id] = dict()
        predictions[dataset_id] = dict()
        predictions[dataset_id]["models"] = dict()
        losses[dataset_id] = dict()
        losses[dataset_id]["models"] = dict()
        features[dataset_id] = dict()
        features[dataset_id]["models"] = dict()

        # Iterate over all images
        for image_id in union_image_ids[dataset_id]:

            common_image = True
            skip_image = False
            tps = []
            # Iterate over all models 
            for i, model_id in enumerate(metrics[dataset_id]["models"].keys()):
                metric = metrics[dataset_id]["models"][model_id]
                if image_id in metric["images"]:
                    size = metric["images"][image_id]["size"]
                    if len(size) < 3: # gray scale
                        skip_image = True
                    if skip_image:
                        metric["images"].pop(image_id)
                        continue
                else:
                    common_image = False
                    continue
                iou = np.array(metric["images"][image_id]["iou"])
                # Compute true positives in each image
                above_iou_threshold = iou >= iou_threshold
                metric["images"][image_id]["tp"] = above_iou_threshold.astype(int)
                tps.append(metric["images"][image_id]["tp"])

            if skip_image:
                continue
            if common_image:
                # Find common true positive targets
                common_tps[dataset_id][image_id] = np.all(np.array(tps), axis=0)
            else:
                if loss_on_common_tps:
                    continue

            # Calculate losses for each model
            for i, model_id in enumerate(metrics[dataset_id]["models"].keys()):
                metric = metrics[dataset_id]["models"][model_id]
                size = metric["images"][image_id]["size"]
                target = np.array(metric["images"][image_id]["target"])
                prediction = np.array(metric["images"][image_id]["prediction"])
                # Mask for true positive
                if loss_on_common_tps:
                    tp = common_tps[dataset_id][image_id]
                else:
                    tp = metric["images"][image_id]["tp"]
                if not np.any(tp):
                    continue
                # Compute loss on true positive
                loss = compute_l2_loss(
                    target[tp][:, :4], 
                    prediction[tp][:, :4], 
                    format="xyxy", 
                    normalize=normalize_coordinates, 
                    mode=None, 
                    width=size[2], 
                    height=size[1],
                )
                # Record loss
                if model_id not in losses[dataset_id]["models"]:
                    losses[dataset_id]["models"][model_id] = []
                losses[dataset_id]["models"][model_id].append(loss)
                # Record true positive prediction
                if model_id not in predictions[dataset_id]["models"]:
                    predictions[dataset_id]["models"][model_id] = []
                pred = prediction[tp][:, :4]
                if normalize_coordinates:
                    pred[:, 0], pred[:, 2] = pred[:, 0]/size[2], pred[:, 2]/size[2]
                    pred[:, 1], pred[:, 3] = pred[:, 1]/size[1], pred[:, 3]/size[1]
                predictions[dataset_id]["models"][model_id].append(pred)

                # Record feature of true positive prediction
                if "feature" in metric["images"][image_id].keys():
                    feature = metric["images"][image_id]["feature"]
                    feature = [f for f, v in zip(feature, tp) if v]
                    if model_id not in features[dataset_id]["models"]:
                        features[dataset_id]["models"][model_id] = []
                    features[dataset_id]["models"][model_id].append(feature)

        # Save common true positives for each dataset
        metrics[dataset_id]["iou_threshold"] = iou_threshold
        metrics[dataset_id]["common_tps"] = common_tps[dataset_id]
        metrics[dataset_id]["losses"] = dict()
        metrics[dataset_id]["losses"]["models"] = losses[dataset_id]["models"]
        metrics[dataset_id]["predictions"] = dict()
        metrics[dataset_id]["predictions"]["models"] = predictions[dataset_id]["models"]
        metrics[dataset_id]["features"] = dict()
        metrics[dataset_id]["features"]["models"] = features[dataset_id]["models"]
        # metrics[dataset_id]["predictions"]["center"] = (0.4671 + 0.4433) / 2
        metrics[dataset_id]["predictions"]["center"] = (0.4511 + 0.4598) / 2

    return metrics


def save_metrics(metrics, directory, experiment_id, iou_threshold, verbose=True):
    """
    """
    if verbose:
        # Iterate over all datasets
        for dataset_id in metrics.keys():
            print("On {}, iou {}".format(dataset_id, metrics[dataset_id]["iou_threshold"]))
            common_tps = metrics[dataset_id]["common_tps"]
            for i, model_id in enumerate(metrics[dataset_id]["models"].keys()):
                loss = np.vstack(metrics[dataset_id]["losses"]["models"][model_id])
                prediction = np.vstack(metrics[dataset_id]["predictions"]["models"][model_id])
                print("{} images {}, tps/common {}/{}, pred mean/var {:.4f}/{:.4f}, l2 avg {:.8f}, {}".format(
                    model_id.ljust(max_len_model_id), 
                    len(metrics[dataset_id]["models"][model_id]["images"].keys()), 
                    sum([np.sum(v["tp"]) for k, v in metrics[dataset_id]["models"][model_id]["images"].items()]), 
                    sum([np.sum(v) for k, v in common_tps.items()]), 
                    np.mean(np.mean(prediction, axis=0)),
                    np.mean(np.var(prediction, axis=0)),
                    np.mean(loss),
                    np.mean(loss, axis=0),
                    )
                )

    # Save metrics to JSON
    save_filename = "{}_iou{}.json".format(experiment_id, iou_threshold)
    save_path = os.path.join(directory, save_filename)
    with open(save_path, 'w') as f:
        json.dump(metrics, f, cls=JsonNumericEncoder)
    print("{} Metrics saved to {}.".format(get_datetime(), save_path))
           

def run_experiment(args):
    """
    """
    directory = args.directory
    filenames = args.filenames
    experiment_id = args.experiment_id
    iou_threshold = args.iou_threshold
    loss_on_common_tps = args.loss_on_common_tps
    normalize_coordinates = args.normalize_coordinates

    predictions, dataset_ids, model_ids, union_image_ids = load_predictions(directory, filenames)
    metrics = summarize_metrics(
        predictions, 
        dataset_ids,
        model_ids,
        union_image_ids,
        iou_threshold,
        loss_on_common_tps=loss_on_common_tps,        
        normalize_coordinates=normalize_coordinates,
    )
    save_metrics(metrics, directory, experiment_id, iou_threshold)


def run():
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, help="Directory to prediction files.")
    parser.add_argument("--filenames", nargs="+", default=[None], type=pathlib.Path, help="Filenames of prediction files.")
    parser.add_argument("--experiment-id", type=pathlib.Path, help="Experiment identifier.")
    parser.add_argument("--iou-threshold", default=0.5, type=float, help="IOU threshold for determining true positive.")
    parser.add_argument("--loss-on-common-tps", action="store_true", help="If set, compute loss only on common detected targets.")
    parser.add_argument("--normalize-coordinates", action="store_true", help="If set, normalize bbox coordinates before computing l2 loss.")
    args = parser.parse_args()

    run_experiment(args)


def main():
    run()


if __name__ == "__main__":
    main()
