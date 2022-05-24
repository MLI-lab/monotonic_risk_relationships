from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import json
import math
import sys
import argparse
import random
from itertools import compress

from utilities import get_datetime

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

colors = [
    (0.8, 0.25, 0.33),  # brickred
    (0.56, 0.74, 0.56), # darkgreen
    (0.27, 0.51, 0.71), # steelblue
    (0.79, 0.0, 0.09), # harvardcrimson
    (0.85, 0.11, 0.51), # vividcerise
    (0.95, 0.52, 0.0), # tangerine
    (0.6, 0.73, 0.45), # olivine
    (0.58, 0.77, 0.45), # pistachio
    (0.13, 0.55, 0.13), # forestgreen
    (0.0, 0.34, 0.25), # sacramentostategreen
    (0.63, 0.36, 0.94), # veronica
]


def get_parameters(cfg):
    """
    """
    snr, gamma = cfg["snr"], cfg["gamma"]

    var_p = 1.0 / snr * np.ones_like(gamma)
    var_q = var_p * gamma 
    var_p = dict(zip(gamma, var_p))
    var_q = dict(zip(gamma, var_q))

    return var_p, var_q 


def get_denoising_experiment_data_points(cfg, var_p, var_q):
    """
    risk = E||Uc - x_h||_2^2
    """
    d, ds, snr, gamma, affinity, lmbd = cfg["d"], cfg["ds"], cfg["snr"], cfg["gamma"], cfg["affinity"], cfg["lmbd"]
    risk_p = {}
    risk_q = {}

    # get data points
    for k1 in affinity:
        for k2 in gamma:
           alpha = 1 / (1 + var_p[k2] + lmbd)
           risk_p[(k1, k2)] = (ds * alpha * (alpha - 2) + ds + alpha**2 * ds * var_p[k2]) / ds
           risk_q[(k1, k2)] = (k1 * ds * alpha * (alpha - 2) + ds + alpha**2 * ds * var_q[k2]) / ds

    return risk_p, risk_q, affinity, gamma


def subplot(cfg, risk_p, risk_q, affinity, gamma, n_subplots, fig=None, axes=None):
    """
    """
    d, ds, snr, lmbd, subplot_idx, add_legend, title = cfg["d"], cfg["ds"], cfg["snr"], cfg["lmbd"], cfg["subplot_idx"], cfg["add_legend"], cfg["title"]
    curve_label = r"$affinity = {:.1f}, \gamma = {:.1f}$"
    point_label = r"$\lambda = {:.1f}$"
    identity_label = r"$y = x$"
    xlabel = r"$\ell_2$ risk on $P$"
    ylabel = r"$\ell_2$ risk on $Q$"
    label_fontsize = 15
    label_fontweight = "bold"
    tick_labelsize = 15
    legend_fontsize = "x-large"
    lgd_bbox_to_anchor = [1.05, 0.0]
    lgd_loc = "lower left"
    if fig is None or axes is None:
        fig, axes = plt.subplots(1, n_subplots)

    if n_subplots == 1:
        ax = axes
    else:
        ax = axes[subplot_idx]

    count = 0
    for k1 in affinity:
        for k2 in gamma:
            ax.plot(risk_p[(k1, k2)], risk_q[(k1, k2)], label=curve_label.format(k1, k2), linestyle='-', linewidth=2.0, marker="o", color=colors[count])
            count += 1
    risk_low = min(risk_p[(k1, k2)])
    risk_high = max(risk_p[(k1, k2)])
    ax.plot((risk_low, risk_high), (risk_low, risk_high), label=identity_label, linestyle='--', linewidth=2.0, marker="")
    ax.set_xlabel(xlabel, fontweight=label_fontweight, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontweight=label_fontweight, fontsize=label_fontsize)
    ax.tick_params(axis="both", which="major", labelsize=tick_labelsize)
    ax.grid()
    if add_legend:
        ax.legend(bbox_to_anchor=lgd_bbox_to_anchor, loc=lgd_loc, ncol=1, shadow=True, fontsize=legend_fontsize)
    ax.title.set_text(title)

    return fig, axes


def run_denoising_experiments():
    """
    """
    experiment_configs = {
        # "low_snr_noise_variance_change": {
        #     "d": 100,
        #     "ds": 20,
        #     "snr": 1,
        #     # "gamma": np.array([0.4, 0.5, 0.67, 1.0, 1.5, 2.0, 2.5]),
        #     "gamma": np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]),
        #     "affinity": np.array([1.0]),
        #     "lmbd": np.arange(0, 15, 0.5),
        #     "subplot_idx": 0,
        #     "add_legend": False,
        #     "title": r"SNR = 1, noise change",
        # },
        "low_snr_subspace_change": {
            "d": 1000,
            "ds": 200,
            "snr": 1,
            "gamma": np.array([1.0]),
            "affinity": np.array([0.0, 0.5, 1.0]), #np.arange(0, 1.2, 0.2),
            "lmbd": np.arange(0, 15, 0.5),
            "subplot_idx": 0, # 3,
            "add_legend": False,
            "title": r"SNR = 1, subspace change",
        },
        # "high_snr_noise_variance_change": {
        #     "d": 100,
        #     "ds": 20,
        #     "snr": 100,
        #     # "gamma": np.array([0.4, 0.5, 0.67, 1.0, 1.5, 2.0, 2.5]),
        #     "gamma": np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]),
        #     "affinity": np.array([1.0]),
        #     "lmbd": np.arange(0, 15, 0.5),
        #     "subplot_idx": 1,
        #     "add_legend": True,
        #     "title": r"SNR = 100, noise change",
        # },
        "high_snr_subspace_change": {
            "d": 1000,
            "ds": 200,
            "snr": 100,
            "gamma": np.array([1.0]),
            "affinity": np.array([0.0, 0.5, 1.0]), #np.arange(0, 1.2, 0.2),
            "lmbd": np.arange(0, 15, 0.5),
            "subplot_idx": 1, # 4,
            "add_legend": True,
            "title": r"SNR = 100, subspace change",
        },
    }

    n_subplots = len(experiment_configs)
    fig, axes = None, None
    for i, experiment_id in enumerate(experiment_configs):
        cfg = experiment_configs[experiment_id]
        var_p, var_q = get_parameters(cfg)
        risk_p, risk_q, affinity, gamma = get_denoising_experiment_data_points(cfg, var_p, var_q)
        fig, axes = subplot(cfg, risk_p, risk_q, affinity, gamma, n_subplots, fig=fig, axes=axes)
    plt.show()
    fig.savefig("fig.png", bbox_inches="tight")


def run_experiments(args):
    """
    """
    seed = args.seed
    meta_experiment_id = args.meta_experiment_id

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)

    if meta_experiment_id == "denoising":
        run_denoising_experiments()
    else:
        raise ValueError("'meta_experiment_id' should be either 'denoising' or 'compressed_sensing'.")


def run():
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default="1009", help="Random seed.")
    parser.add_argument("--meta-experiment-id", type=str, default="loss", help="Input 'denoising'.")
    args = parser.parse_args()

    run_experiments(args)


def main():
    run()

    
if __name__ == "__main__":
    main()



