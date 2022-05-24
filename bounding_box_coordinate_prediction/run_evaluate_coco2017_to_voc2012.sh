#!/bin/bash

python -u run_experiment.py \
    --experiment-id experiment_coco2017_to_voc2012 \
    --model-ids \
        yolov5n \
	yolov5s \
	yolov5m \
	yolov5l \
	yolov5x \
        fasterrcnn_resnet50_fpn \
        fasterrcnn_mobilenet_v3_large_fpn \
        fasterrcnn_mobilenet_v3_large_320_fpn \
        retinanet_resnet50_fpn \
        ssd300_vgg16 \
        ssdlite320_mobilenet_v3_large \
        maskrcnn_resnet50_fpn \
        keypointrcnn_resnet50_fpn \
    --dataset-ids coco_val2017 voc2012 \
# Substitute the following lines to get features of YOLOv5s
#     --model-ids yolov5s \
#     --dataset-ids coco_val2017 voc2012 \
#     --save-features \
