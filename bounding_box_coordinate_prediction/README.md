## Figure 1
Download YOLOv5 model checkpoints manually from https://github.com/ultralytics/yolov5/releases and save yolov5n.pt, yolov5s.pt, yolov5m.pt, yolov5l.pt, yolov5x.pt to the current directory. These models are loaded later creating YOLOv5 instances (using a modified source code) where feature vectors are recorded.

Run inference with specified models.
```
sh run_evaluate_coco2017_to_voc2012.sh
```

Summarize inference output and calculate metrics.
```
sh run_summarize_metrics.sh
```

### Figure 1(middle, right)
Plot squared error losses.
```
sh run_plot.sh
```

### Figure 1(middle, right)

Run inference with YOLOv5s.
```
sh run_evaluate_coco2017_to_voc2012.sh
```
Specify `--model-ids yolov5s` and `--save-features` in `run_evaluate_coco2017_to_voc2012.sh` as
```
    --model-ids yolov5s \
    --save-features \
```

Summarize inference output and calculate metrics again.
```
sh run_summarize_metrics.sh
```
Specify `--filenames` in `run_summarize_metrics.sh` as
```
    --filenames \
        predictions_coco_val2017_yolov5s.json \
        predictions_voc2012_yolov5s.json \
```

Plot singular values and subspace similarity.
```
sh run_plot.sh
```
Specify `--mode pca` in `run_plot.sh` as
```
    --mode pca
```

