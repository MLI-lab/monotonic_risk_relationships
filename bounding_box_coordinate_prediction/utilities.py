import torch
import json
import numpy as np
import datetime as dt


def get_datetime():
    """
    """
    return dt.datetime.now().replace(microsecond=0)


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def compute_iou(target, prediction, format="xywh", eps=1e-7):
    if format == "xywh":
        target = xywh2xyxy(target)
        prediction = xywh2xyxy(prediction)
    target = target[:, None, :]
    prediction = prediction[None, :, :]
    top_left_min = np.where(target[:, :, :2] <= prediction[:, :, :2], target[:, :, :2], prediction[:, :, :2])
    top_left_max = np.where(target[:, :, :2] >= prediction[:, :, :2], target[:, :, :2], prediction[:, :, :2])
    bottom_right_min = np.where(target[:, :, 2:] <= prediction[:, :, 2:], target[:, :, 2:], prediction[:, :, 2:])
    bottom_right_max = np.where(target[:, :, 2:] >= prediction[:, :, 2:], target[:, :, 2:], prediction[:, :, 2:])
    intersection = np.clip(bottom_right_min - top_left_max, 0, None)
    intersection = intersection[:, :, 0] * intersection[:, :, 1]
    union = bottom_right_max - top_left_min
    union = union[:, :, 0] * union[:, :, 1]
    iou = intersection / (union + eps)
    return iou


def compute_l2_loss(target, prediction, format="xywh", normalize=False, mode="pairwise", width=640, height=640):
    if format == "xywh":
        target = xywh2xyxy(target)
        prediction = xywh2xyxy(prediction)
    if normalize:
        target, prediction = np.copy(target), np.copy(prediction)
        target[:, 0], target[:, 2] = target[:, 0]/width, target[:, 2]/width
        target[:, 1], target[:, 3] = target[:, 1]/height, target[:, 3]/height
        prediction[:, 0], prediction[:, 2] = prediction[:, 0]/width, prediction[:, 2]/width
        prediction[:, 1], prediction[:, 3] = prediction[:, 1]/height, prediction[:, 3]/height
    if mode == "pairwise":
        target = target[:, None, :]
        prediction = prediction[None, :, :]
    l2_loss = (target - prediction) ** 2
    return l2_loss


class JsonNumericEncoder(json.JSONEncoder):
    """
    Json encoder for numpy data types.
    """
    def default(self, obj):
        """
        Serialize object. This method overrides the base method
        Args:
            obj: object to serialize
        Returns:
            serialized object
        """
        if isinstance(obj, (int, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, \
                np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (float, np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def JsonNumericDecoder(ordered_pairs):
    """
    Json decoder for non-string data types.
    """
    SPECIAL = {
        "true": True,
        "false": False,
        "null": None,
    }
    result = {}
    for key, value in ordered_pairs:
        if key in SPECIAL:
            key = SPECIAL[key]
        else:
            for numeric in [int, float]:
                try:
                    key = numeric(key)
                except ValueError:
                    continue
                else:
                    break
        result[key] = value
    return result
