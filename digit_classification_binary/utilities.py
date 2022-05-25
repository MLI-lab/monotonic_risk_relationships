import sys
import torch
import json
import numpy as np
import datetime as dt

from operator import attrgetter


def get_datetime():
    """
    """
    return dt.datetime.now().replace(microsecond=0)


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


