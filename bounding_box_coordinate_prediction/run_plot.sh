#!/bin/bash

python3 -u plot.py \
    --path ./experiment_coco2017_to_voc2012/metrics_features_coco2017_to_voc2012_iou0.2.json \
    --mode loss \
# Substitute the following to plot singular values and subspace similarity
#     --mode pca \

