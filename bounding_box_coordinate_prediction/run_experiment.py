import os
import torch
import torchvision
import json
import argparse
import pathlib
import numpy as np
import xml.etree.ElementTree as ET

from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from utilities import compute_iou, compute_l2_loss
from utilities import coco80_to_coco91_class, xyxy2xywh, xywh2xyxy
from utilities import get_datetime
from utilities import JsonNumericEncoder


coco80_to_coco91_map = coco80_to_coco91_class()
model_id_group1 = ["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"]
model_id_group2 = ["fasterrcnn_resnet50_fpn", "fasterrcnn_mobilenet_v3_large_fpn",
                 "fasterrcnn_mobilenet_v3_large_320_fpn", "retinanet_resnet50_fpn",
                 "ssd300_vgg16", "ssdlite320_mobilenet_v3_large",
                 "maskrcnn_resnet50_fpn", "keypointrcnn_resnet50_fpn"]


class COCOBboxDataset(Dataset):
    """
    """
    def __init__(self, image_directory, label_path):
        """
        """
        self.image_directory = image_directory
        self.label_path = label_path
        self.image_filename_width = 12
        self.excluded_image_ids = ["130465", "141671"]
        self.excluded_image_ids = []
        self.read_targets()
        self.image_ids = sorted(self.targets.keys())

    def read_targets(self):
        """
        """
        with open(self.label_path, 'r') as f:
            instances = json.load(f)
        labels = instances["annotations"]
        self.targets = dict()
        for ann in labels:
            image_id = str(ann["image_id"])
            if image_id in self.excluded_image_ids:
                continue
            if image_id not in self.targets and image_id not in self.excluded_image_ids:
                self.targets[image_id] = []
            self.targets[image_id].append([*ann["bbox"], ann["category_id"]])
        for image_id in self.targets:
            self.targets[image_id] = np.array(self.targets[image_id])

    def __len__(self):
        """
        """
        return len(self.targets)

    def __getitem__(self, i):
        """
        """
        sample = dict()
        image_id = self.image_ids[i]
        image_filename = "{}.jpg".format(str(image_id).zfill(self.image_filename_width))
        try:
            image = Image.open(os.path.join(self.image_directory, image_filename))
            image = np.array(image)
            image = image.transpose((2, 0, 1))
        except:
            pass
            # raise ValueError("Error loading image {}".format(image_filename))
        sample["image_id"] = image_id
        sample["image"] = image
        sample["target"] = self.targets[image_id]
        return sample


class VOCBboxDataset(Dataset):
    """
    """
    def __init__(self, image_directory, label_directory):
        """
        """
        self.image_directory = image_directory
        self.label_directory = label_directory
        self.categories = [
            "background",
            "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor",
        ]
        self.category_to_ind = dict(zip(self.categories, range(len(self.categories))))
        self.read_targets()
        self.image_ids = sorted(self.targets.keys())
        
    def read_targets(self):
        """
        """
        filenames = os.listdir(self.label_directory)
        self.targets = dict()
        for filename in filenames:
            image_id = filename[:-4]
            path = os.path.join(self.label_directory, filename)
            tree = ET.parse(path)
            root = tree.getroot()
            for bbox in root.iter("object"):
                category_id = self.category_to_ind[bbox.find("name").text.lower().strip()]
                x1 = float(bbox.find("bndbox/xmin").text)
                x2 = float(bbox.find("bndbox/xmax").text)
                y1 = float(bbox.find("bndbox/ymin").text)
                y2 = float(bbox.find("bndbox/ymax").text)
                if image_id not in self.targets:
                    self.targets[image_id] = []
                self.targets[image_id].append([x1, y1, x2, y2, category_id])
        for image_id in self.targets:
            self.targets[image_id] = np.array(self.targets[image_id])

    def __len__(self):
        """
        """
        return len(self.targets)

    def __getitem__(self, i):
        """
        """
        sample = dict()
        image_id = self.image_ids[i]
        image_filename = "{}.jpg".format(image_id)
        try:
            image = Image.open(os.path.join(self.image_directory, image_filename))
            image = np.array(image)
            image = image.transpose((2, 0, 1))
        except:
            pass
            # raise ValueError("Error loading image {}".format(image_filename))
        sample["image_id"] = image_id
        sample["image"] = image
        sample["target"] = self.targets[image_id]
        return sample


def get_features(model, predictions):
    """
    """
    num_anchors = model.model.model[-1].na
    indices = predictions.indices # [Tensor(), ...]
    features = [[]] * len(indices)
    feature_maps = model.model.feature_maps # {layer_number: Tensor(bs, nc, ny, nx), ...}
    layers = sorted(list(feature_maps.keys())) # [layer_number, ...]
    feature_maps = [feature_maps[l] for l in layers]

    for feature_map in feature_maps:
        if feature_map.shape[0] != len(indices):
            raise ValueError("Dimension not equal: feature_map.shape[0]({}) and len(indices)({})".format(feature_map.shape[0], len(indices)))

    feature_map_shapes = [fm.shape for fm in feature_maps] # [Tensor(bs, nc, ny, nx), ...]
    feature_map_spatial_sizes = torch.tensor([s[-2] * s[-1] for s in feature_map_shapes])
    feature_map_sizes = num_anchors * feature_map_spatial_sizes # [na*ny*nx, ...]
    cum_feature_map_sizes = torch.cumsum(feature_map_sizes, 0)

    # Iterate over predictions in a batch
    for j, index in enumerate(indices):
        # Iterate over predicted bounding boxes from a single image input
        for ind in index:
            # Calculate which feature map and cell a predicted bounding box corresponds to
            feature_map_index = torch.where(ind < cum_feature_map_sizes)[0][0]
            i = (ind - cum_feature_map_sizes[feature_map_index-1]) % feature_map_spatial_sizes[feature_map_index]
            y_index = torch.div(i, feature_map_shapes[feature_map_index][-1], rounding_mode="floor").int()
            x_index = (i % feature_map_shapes[feature_map_index][-1]).int()
            # Extract features
            features[j].append(feature_maps[feature_map_index][j, :, y_index, x_index])
    return features


def _infer(sample, model, model_id, save_features=False):
    """
    """
    image_ids = sample["image_id"]
    images = sample["image"]
    sizes = [np.array(image.shape) for image in images]
    targets = [xywh2xyxy(target.numpy()) for target in sample["target"]]
    predictions = []
    features = []

    for i, image in enumerate(images):
        if len(image.shape) < 3: # gray scale
            sizes[i] = None
            predictions.append(None)
            features.append(None)
            continue
        if model_id in model_id_group1:
            image = image.float().numpy()[0]
        elif model_id in model_id_group2:
            image = image.float() / 255.
        else:
            raise ValueError("Model id {} not in predefined groups.".format(model_id))

        try:
            detection = model(image) # detection.pred: [Tensor(n, 6), ...]
        except:
            continue
        feature = None
        if save_features:
            feature = get_features(model, detection) # [[Tensor(nc), ...], ...]
            feature = [f.detach().cpu().numpy() for f in feature[0]]
            
        if model_id in model_id_group1:
            detection = detection.xyxy[0].detach().cpu().numpy()
            detection[:, 5] = [coco80_to_coco91_map[y] for y in detection[:, 5].astype(int)]
        elif model_id in model_id_group2:
            predicted_bbox = detection[0]["boxes"].detach().cpu().numpy()
            predicted_label = detection[0]["labels"].detach().cpu().numpy()
            predicted_confidence = detection[0]["scores"].detach().cpu().numpy()
            detection = np.hstack((predicted_bbox, predicted_confidence[:, None], predicted_label[:, None]))
        else:
            raise ValueError("Model id {} not in predefined groups.".format(model_id))

        if feature is None:
            feature = [None] * detection.shape[0]

        predictions.append(detection)
        features.append(feature)

    return image_ids, sizes, targets, predictions, features


def infer(model, model_id, dataloader, num_samples=None, use_tqdm=False, save_features=False):
    """
    """
    output = {}
    # with tqdm(total=len(dataloader) if num_samples is None else num_samples) as pbar:
    if use_tqdm:
        pbar = tqdm(total=len(dataloader) if num_samples is None else num_samples)
    for i, sample in enumerate(dataloader):
        image_ids, sizes, targets, predictions, features = _infer(sample, model, model_id, save_features=save_features)
        for image_id, size, target, prediction, feature in zip(image_ids, sizes, targets, predictions, features):
            if size is None:
                continue
            output[image_id] = {
                "size": size,
                "target": target,
                "prediction": prediction,
                "feature": feature,
            }
        if num_samples is not None and i >= num_samples-1:
            break
        if use_tqdm:
            pbar.update(1)
    if use_tqdm:
        pbar.close()

    return output 


def postprocess(predictions, model_id, dataset_id, target_category=None, prediction_category=None, normalize_coordinates=False, iou_threshold=0.5):
    """
    """
    post_predictions = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "images": dict(),
    }
    # with tqdm(total=len(predictions)) as pbar:
    for image_id, result in predictions.items():
        size = result["size"]
        target = result["target"]
        prediction = result["prediction"] # array of shape (n, 5+1)
        feature = result["feature"] # [Tensor(nc), ...]
        if target.size == 0 or prediction.size == 0:
            continue

        if target_category is not None and prediction_category is not None:
            target_category_instances = np.where(target[:, -1] == target_category)[0]
            prediction_category_instances = np.where(prediction[:, -1] == prediction_category)[0]
            if target_category_instances.size == 0 or prediction_category_instances.size == 0:
                continue
            target_of_interest = target[target_category_instances]
            prediction_of_interest = prediction[prediction_category_instances]
            feature_of_interest = [feature[i] for i in prediction_category_instances]
        else:
            target_of_interest = target
            prediction_of_interest = prediction

        iou = compute_iou(target_of_interest[:, :4], prediction_of_interest[:, :4], format="xyxy") # shape (n_targets, n_predictions)
        indices_max_iou = np.argmax(iou, axis=1) # find prediction of max iou with each target
        max_iou = np.max(iou, axis=1)
        post_predictions["images"][image_id] = {
            "size": size,
            "iou": max_iou, # [iou, ...]
            "target": target_of_interest, # [[x1, y1, x2, y2, category], ...]
            "prediction": prediction_of_interest[indices_max_iou], # [[x1, y1, x2, y2, confidence, category], ...]
            "feature": [feature_of_interest[i] for i in indices_max_iou], # [Tensor(nc), ...]
        }
        # pbar.update(1)
    return post_predictions


def save_predictions(metrics, save_path):
    """
    """
    with open(save_path, 'w') as f:
        json.dump(metrics, f, cls=JsonNumericEncoder)


def prepare_model(model_id):
    """
    """
    if model_id in model_id_group1:
        # model = torch.hub.load("ultralytics/yolov5", model_id, force_reload=True)
        # Load YOLOv5 manually using a modified model where feature vectors are recorded
        # Check lines with comment "(added lines)" in models/yolo.py, models/common.py and utils/general.py under ./yolo
        import yaml
        import yolov5
        from yolov5.models.yolo import Model
        from yolov5.models.common import AutoShape
        with open("./yolov5/models/{}.yaml".format(model_id), 'r') as f:
            cfg = yaml.safe_load(f)
        model = AutoShape(Model(cfg, ch=3, nc=80, anchors=3))
        state_dict = torch.load("./{}.pt".format(model_id))["model"].state_dict()
        for k in sorted(state_dict.keys()):
            new_k = "model." + k
            state_dict[new_k] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    if model_id in model_id_group2:
        model = eval("torchvision.models.detection.{}(pretrained=True)".format(model_id))
        model.eval()

    return model


def configure_dataset(dataset_id):
    """
    """
    if dataset_id == "coco_val2017":
        # COCO 2017
        cfg = {
            "image_directory": "/workspace/yosemite_root/media/hdd1/coco_val2017/coco/images",
            "label_directory": "/workspace/yosemite_root/media/hdd1/coco_val2017/coco/annotations/instances_val2017.json",
            "dataset_class": COCOBboxDataset,
            "dataset_id": "coco_val2017",
            "num_samples": None,
            "target_category": 1, # person 1
            "prediction_category": 1, # person 1
            "batch_size": 1,
            "shuffle": False,
        }
    elif dataset_id == "voc2012":
        # VOC 2012
        cfg = {
             "image_directory": "/workspace/yosemite_root/media/hdd1/voc2012/JPEGImages",
             "label_directory": "/workspace/yosemite_root/media/hdd1/voc2012/Annotations",
             "dataset_class": VOCBboxDataset,
             "dataset_id": "voc2012",
             "num_samples": None,
             "target_category": 15, # person 1
             "prediction_category": 1, # person 1
             "batch_size": 1,
             "shuffle": False,
        }

    return cfg


def create_dataloader(dataset_config):
    """
    """
    image_directory = dataset_config["image_directory"]
    label_directory = dataset_config["label_directory"]
    dataset_class = dataset_config["dataset_class"]
    batch_size = dataset_config["batch_size"]
    shuffle = dataset_config["shuffle"]

    dataset = dataset_class(image_directory, label_directory)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print("{} {} and dataloader of length {} created".format(get_datetime(), dataset_class, len(dataloader)))

    return dataloader


def evaluate(dataloader, dataset_config, model_id, experiment_id, save_features=False):
    """
    """
    dataset_id = dataset_config["dataset_id"]
    num_samples = dataset_config["num_samples"]
    target_category = dataset_config["target_category"]
    prediction_category = dataset_config["prediction_category"]

    save_predictions_path = os.path.join("./{}".format(experiment_id), "predictions_{}_{}.json".format(dataset_id, model_id))
    print("\n\n{} Evaluating model {}...".format(get_datetime(), model_id))
    model = prepare_model(model_id)
    print("{} Inferring with {}...".format(get_datetime(), model_id))
    predictions = infer(model, model_id, dataloader, num_samples=num_samples, save_features=save_features)
    print("{} Postprocessing predictions of {}...".format(get_datetime(), model_id))
    predictions = postprocess(predictions, model_id, dataset_id, target_category, prediction_category)
    save_predictions(predictions, save_predictions_path)
    print("{} Predictions saved to {}.".format(get_datetime(), save_predictions_path))


def run_experiments(args):
    """
    """
    model_ids = args.model_ids
    dataset_ids = args.dataset_ids
    experiment_id = args.experiment_id
    save_features = args.save_features

    for dataset_id in dataset_ids:
        print("\n\n\n{} Evaluating models on {}...".format(get_datetime(), dataset_id))
        dataset_config = configure_dataset(dataset_id)
        dataloader = create_dataloader(dataset_config)
        for model_id in model_ids:
            evaluate(dataloader, dataset_config, model_id, experiment_id, save_features=save_features)
    

def run():
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ids", nargs="+", default=[None], type=str, help="Model identification strings.")
    parser.add_argument("--dataset-ids", nargs="+", default=[None], type=str, help="Dataset identification strings.")
    parser.add_argument("--experiment-id", type=str, help="Experiment identification string.")
    parser.add_argument('--save-features', action='store_true', help="Save features to JSON.")
    args = parser.parse_args()

    run_experiments(args)


def main():
    run()


if __name__ == "__main__":
    main()
