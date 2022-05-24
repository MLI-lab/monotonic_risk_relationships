#!/bin/bash

python3 -u summarize_metrics.py \
    --experiment-id metrics_coco2017_to_voc2012 \
    --iou-threshold 0.2 \
    --loss-on-common-tps \
    --normalize-coordinates \
    --directory ./experiment_coco2017_to_voc2012 \
    --filenames \
        predictions_coco_val2017_yolov5s.json \
        predictions_voc2012_yolov5s.json \
        predictions_coco_val2017_yolov5n.json \
        predictions_coco_val2017_yolov5s.json \
        predictions_coco_val2017_yolov5m.json \
        predictions_coco_val2017_yolov5l.json \
        predictions_coco_val2017_yolov5x.json \
        predictions_coco_val2017_fasterrcnn_resnet50_fpn.json \
        predictions_coco_val2017_fasterrcnn_mobilenet_v3_large_fpn.json \
        predictions_coco_val2017_fasterrcnn_mobilenet_v3_large_320_fpn.json \
        predictions_coco_val2017_retinanet_resnet50_fpn.json \
        predictions_coco_val2017_ssd300_vgg16.json \
        predictions_coco_val2017_ssdlite320_mobilenet_v3_large.json \
        predictions_coco_val2017_maskrcnn_resnet50_fpn.json \
        predictions_coco_val2017_keypointrcnn_resnet50_fpn.json \
        predictions_voc2012_yolov5n.json \
        predictions_voc2012_yolov5s.json \
        predictions_voc2012_yolov5m.json \
        predictions_voc2012_yolov5l.json \
        predictions_voc2012_yolov5x.json \
        predictions_voc2012_fasterrcnn_resnet50_fpn.json \
        predictions_voc2012_fasterrcnn_mobilenet_v3_large_fpn.json \
        predictions_voc2012_fasterrcnn_mobilenet_v3_large_320_fpn.json \
        predictions_voc2012_fasterrcnn_resnet50_fpn.json \
        predictions_voc2012_ssd300_vgg16.json \
        predictions_voc2012_ssdlite320_mobilenet_v3_large.json \
        predictions_voc2012_maskrcnn_resnet50_fpn.json \
        predictions_voc2012_keypointrcnn_resnet50_fpn.json \
# To get the metric file containing features of YOLOv5s, 
# substitute the following lines with output from executing "sh run_evaluate_coco2017_to_voc2012.sh" for yolov5s, where "--save-features" is specified
#     --filenames \
#         predictions_coco_val2017_yolov5s.json \
#         predictions_voc2012_yolov5s.json \
