#!/bin/bash

python -u run_experiment.py \
    --seed 1009 \
    --mode evaluate \
    --gpu 3 \
    --dataset-ids mnist ardis \
    --model-ids \
        alexnet \
        vgg11 \
        vgg16 \
        resnet18 \
        resnet50 \
        densenet121 \
        densenet161 \
    --experiment-id experiment_mnist_to_ardis_binary_v1 \
    --data-configs \
        ./experiment_mnist_to_ardis_binary_v1/mnist_data_config.yaml \
        ./experiment_mnist_to_ardis_binary_v1/ardis_data_config.yaml \
    --model-configs \
        ./experiment_mnist_to_ardis_binary_v1/mnist_alexnet_model_config_ckpt_saving_hook.yaml \
        ./experiment_mnist_to_ardis_binary_v1/mnist_vgg_model_config_ckpt_saving_hook.yaml \
        ./experiment_mnist_to_ardis_binary_v1/mnist_vgg_model_config_ckpt_saving_hook.yaml \
        ./experiment_mnist_to_ardis_binary_v1/mnist_resnet_model_config_ckpt_saving_hook.yaml \
        ./experiment_mnist_to_ardis_binary_v1/mnist_resnet_model_config_ckpt_saving_hook.yaml \
        ./experiment_mnist_to_ardis_binary_v1/mnist_densenet_model_config_ckpt_saving_hook.yaml \
        ./experiment_mnist_to_ardis_binary_v1/mnist_densenet_model_config_ckpt_saving_hook.yaml \
