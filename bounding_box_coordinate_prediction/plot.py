import os
import json
import argparse
import pathlib
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.linalg import subspace_angles

from utilities import compute_l2_loss
from utilities import get_datetime


max_len_model_id = 37

def load_metrics(path):
    """
    """
    # Load metrics
    with open(path, 'r') as f:
        metrics = json.load(f)

    return metrics


def print_metrics(metrics):
    """
    """
    losses = dict()
    # Iterate over all datasets
    for dataset_id in metrics.keys():
        print("On {}, iou {}".format(dataset_id, metrics[dataset_id]["iou_threshold"]))
        common_tps = metrics[dataset_id]["common_tps"]
        prediction_center = metrics[dataset_id]["predictions"]["center"]
        losses[dataset_id] = dict()
        for i, model_id in enumerate(metrics[dataset_id]["models"].keys()):
            loss = np.vstack(metrics[dataset_id]["losses"]["models"][model_id])
            prediction = np.vstack(metrics[dataset_id]["predictions"]["models"][model_id])
            print("{} images {}, tps/common {}/{}, pred mean/var {:.4f}/{:.4f} cor {:.4f}, l2 avg {:.8f}, {}".format(
                model_id.ljust(max_len_model_id), 
                len(metrics[dataset_id]["models"][model_id]["images"].keys()), 
                sum([np.sum(v["tp"]) for k, v in metrics[dataset_id]["models"][model_id]["images"].items()]), 
                sum([np.sum(v) for k, v in common_tps.items()]), 
                np.mean(np.mean(prediction, axis=0)),
                np.mean(np.var(prediction, axis=0)),
                np.mean(np.mean((prediction - prediction_center)**2, axis=0)),
                np.mean(loss),
                np.mean(loss, axis=0),
                )
            )
            losses[dataset_id][model_id] = np.mean(loss, axis=0) # np.mean(loss)

    return losses


def plot(metrics):
    """
    """
    dataset_ids = ["coco_val2017", "voc2012"]
    model_ids = [
        "yolov5n",
        "yolov5s",
        "yolov5m",
        "yolov5l",
        "yolov5x",
        "fasterrcnn_resnet50_fpn",
        "fasterrcnn_mobilenet_v3_large_fpn",
        "fasterrcnn_mobilenet_v3_large_320_fpn",
        "retinanet_resnet50_fpn",
        "ssd300_vgg16",
        "ssdlite320_mobilenet_v3_large",
        "maskrcnn_resnet50_fpn",
        "keypointrcnn_resnet50_fpn",
    ]
    
    # shape (n, 4), where each row corresponds to l2 loss on x1, y1, x2 and y2
    losses = print_metrics(metrics)
    # coordinate_index = 1 # 0, 1, 2, 3 for x1, y1, x2, y2
    coordinate_index = None
    if coordinate_index is None: # average mse over x1, y1, x2, y2
        mse_p = np.mean([losses[dataset_ids[0]][model_id] for model_id in model_ids], axis=1)
        mse_q = np.mean([losses[dataset_ids[1]][model_id] for model_id in model_ids], axis=1)
    else:
        mse_p = [losses[dataset_ids[0]][model_id][coordinate_index] for model_id in model_ids]
        mse_q = [losses[dataset_ids[1]][model_id][coordinate_index] for model_id in model_ids]

    x = np.vstack((mse_p, np.ones_like(mse_p))).reshape(2, -1).T
    y = mse_q
    best_fit_params = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    best_fit_slope, best_fit_intercept = best_fit_params[0], best_fit_params[1]
    print("best linear fit: {}".format(best_fit_params))

    labels = model_ids
    xlabel = r"MSE on COCO val 2017"
    ylabel = r"MSE on VOC 2012"
    label_fontsize = 15
    label_fontweight = "bold"
    tick_labelsize = 15
    legend_fontsize = "large"
    
    # xmin, xmax = np.min(mse_p), np.max(mse_p)
    # ymin, ymax = np.min(mse_q), np.max(mse_q)
    margin = None
    # margin = 10*(xmax - xmin) / 2
    xmin, xmax, ymin, ymax = 0.006, 0.020, 0.0, 0.08
    
    fig, ax = plt.subplots(1, figsize=(10, 10))
    for i, label in enumerate(labels):
        ax.scatter(mse_p[i], mse_q[i], label=labels[i])
    ax.plot((0, 1), (0, 1), ":k", label="unity slope")
    ax.plot((0, 1), (best_fit_intercept, best_fit_intercept + best_fit_slope*1), "--k", label="best linear fit")
    if margin is not None:
        ax.set_xlim(xmin-margin, xmax+margin)
        ax.set_ylim(ymin-margin, ymax+margin)
    else:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    ax.set_xlabel(xlabel, fontweight=label_fontweight, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontweight=label_fontweight, fontsize=label_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)
    ax.grid()
    ax.legend(bbox_to_anchor=[1.05, 0.95], loc="upper left", ncol=1, fontsize=legend_fontsize)
    
    plt.show()
    # Save plot
    save_path = "./fig.png"
    fig.savefig(save_path, bbox_inches="tight")


def run_subspace_analysis(metrics):
    """
    """
    risks = print_metrics(metrics)
    dataset_ids = ["coco_val2017", "voc2012"]
    model_ids = [
        "yolov5s",
    ]

    model_id = model_ids[0]
    # features of true positive detections grouped by input image
    _z_p = metrics[dataset_ids[0]]["features"]["models"][model_id]
    _z_q = metrics[dataset_ids[1]]["features"]["models"][model_id]
    # flatten nested list
    _z_p = [np.asarray(zz) for z in _z_p for zz in z]
    _z_q = [np.asarray(zz) for z in _z_q for zz in z]
    # group features by dimension
    z_p = {128: [], 256: [], 512: []}
    z_q = {128: [], 256: [], 512: []}
    for z in _z_p:
        z_p[len(z)].append(z)
    for z in _z_q:
        z_q[len(z)].append(z)
    print("z_p: {}, z_q: {}".format({k: len(z_p[k]) for k in z_p}, {k: len(z_q[k]) for k in z_q}))

    # principal components
    feature_dimension = 512
    pca = PCA()
    pca.fit(z_p[feature_dimension])
    singular_values_p = pca.singular_values_ 
    singular_vectors_p = pca.components_ # shape (n_components, n_features), vectors stored in rows
    explained_variance_ratio_p = pca.explained_variance_ratio_
    cum_explained_variance_ratio_p = np.cumsum(explained_variance_ratio_p)
    pca.fit(z_q[feature_dimension])
    singular_values_q = pca.singular_values_ 
    singular_vectors_q = pca.components_ 
    explained_variance_ratio_q = pca.explained_variance_ratio_
    cum_explained_variance_ratio_q = np.cumsum(explained_variance_ratio_q)

    # principal angles
    truncate_dimension = 100 # 100 # 512
    thetas = []
    cos_theta_rsss = []
    with tqdm(total=truncate_dimension) as pbar:
        for k in range(1, truncate_dimension+1):
            truncated_p = singular_vectors_p[:k, :].T
            truncated_q = singular_vectors_q[:k, :].T
            theta = subspace_angles(truncated_p, truncated_q)
            cos_theta_rss = np.sqrt(np.mean(np.cos(theta) ** 2))
            thetas.append(theta)
            cos_theta_rsss.append(cos_theta_rss)

            pbar.update(1)
    thetas = np.asarray(thetas)
    cos_theta_rsss = np.asarray(cos_theta_rsss)
    xlabel = r"k"
    ylabel = r"Subspace similarity"
    label_fontsize = 15
    label_fontweight = "bold"
    tick_labelsize = 15
    legend_fontsize = "x-large"
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.plot(np.arange(len(theta)), cos_theta_rsss)
    ax.set_xlabel(xlabel, fontweight=label_fontweight, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontweight=label_fontweight, fontsize=label_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)
    ax.grid()
    ax.legend(bbox_to_anchor=[0.98, 0.98], loc="upper right", ncol=1, fontsize=legend_fontsize)
    plt.show()
    # Save plot
    save_path = "./fig_subspace_similarity.png"
    fig.savefig(save_path, bbox_inches="tight")

    print("singular values P: {}...".format(singular_values_p[:50]))
    print("singular values Q: {}...".format(singular_values_q[:50]))
    
    xlabel = r"Index"
    ylabel = r"Singular value"
    label_fontsize = 15
    label_fontweight = "bold"
    tick_labelsize = 15
    legend_fontsize = "x-large"
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.plot(np.arange(len(singular_values_p)), singular_values_p, label="COCO val 2017")
    ax.plot(np.arange(len(singular_values_q)), singular_values_q, label="VOC 2012")
    # ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontweight=label_fontweight, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontweight=label_fontweight, fontsize=label_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)
    ax.grid()
    ax.legend(bbox_to_anchor=[0.98, 0.98], loc="upper right", ncol=1, fontsize=legend_fontsize)
    
    plt.show()
    # Save plot
    save_path = "./fig_singular_values.png"
    fig.savefig(save_path, bbox_inches="tight")
           

def run_experiment(args):
    """
    """
    metrics = load_metrics(args.path)

    if args.mode == "loss":
        plot(metrics)
    elif args.mode == "pca":
        run_subspace_analysis(metrics)
    else:
        raise ValueError("--mode should be either 'loss' or 'pca', but got {}".format(args.mode))


def run():
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=pathlib.Path, help="Path to metric file.")
    parser.add_argument("--mode", type=str, default="loss", help="Path to metric file.")
    args = parser.parse_args()

    run_experiment(args)


def main():
    run()


if __name__ == "__main__":
    main()
