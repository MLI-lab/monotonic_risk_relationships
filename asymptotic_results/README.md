## Figures 2, 3, and 6

To generate Figure 2, run the notebook `simulation.ipynb`.

To generate Figures 3 and 7, run the notebook `classification_theory.ipynb`. This script assumes the existence of `stats_experiment_mnist_to_ardis_binary_v2_s1009.json` before running.

Dependencies are standard scientific computing packages, plus [`tqdm`](https://github.com/tqdm/tqdm) for progressbars and [`cmasher`](https://cmasher.readthedocs.io/) for better colormaps. Both notebooks can be run in only a few minutes. A LaTeX installation is necessary to generate the plots as-is; remove the Matplotlib settings related to this at the top of the notebooks to avoid this dependency.
