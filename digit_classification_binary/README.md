Download [MNIST dataset](http://yann.lecun.com/exdb/mnist/) and [ARDIS dataset IV](https://ardisdataset.github.io/ARDIS/) to `./mnist` and `./ardis` folders.

Train models.
```
sh run_train_mnist_binary.sh
```
Set dataset configuration and model configuration using YAML files under `./experiment_mnist_to_ardis_binary_v1`. Specify random seed `--seed` and GPU `--gpu` in `./run_train_mnist_binary.sh`. Trained model checkpoints will be saved to `./mnist_binary_ckpt_v1`.
To get intermediate checkpoints, set a small number of total training epochs and specify a checkpoint saving hook in model configuration YAML files as
```
num_epochs: 1
ckpt_saving_hook:
    call_freq:
        epoch: 1
        batch: 10
    ckpt_id_type: val_acc
    life:
        epoch: 0
        batch: 300
    ckpt_id:
        - val_acc: 0.5
        - val_acc: 0.6
        - val_acc: 0.7
        - val_acc: 0.8
        - val_acc: 0.9
```

Evaluate models.
```
sh run_evaluate_mnist_to_ardis_binary.sh
```
Specify model checkpoints to evaluate in model configuration YAML files. For example, the following specifies two checkpoints, one trained for full training epochs, and one intermediate checkpoint saved when the validation accuracy first reaches 0.5. `seed` and `ckpt_id` are used to distinguish checkpoints of different random runs.
```
evaluate:
    ckpt:
        vgg11:
            - ckpt_path: ./mnist_binary_ckpt_v1/model_mnist_vgg11_ce_adam0.0001_ep20_bs10_s1009.pt
              seed: 1009
            - ckpt_path: ./mnist_binary_ckpt_v1/model_mnist_vgg11_ce_adam0.0001_ep1_bs10_s1009.pt.val_acc0.5
              seed: 1009
              ckpt_id: val_acc0.5
```
Accuracy and loss statistics will be saved to `./experiment_mnist_to_ardis_binary_v1/stats_experiment_mnist_to_ardis_binary_v1_s1009.json`.

Check `../asymptotic_results/classification_theory.ipynb` for plotting functions.
