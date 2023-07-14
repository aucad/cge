import os
import re
import time
import warnings

import yaml
from typing import Any, Sized

import numpy as np
import pandas as pd


class Utility:
    """Miscellaneous utility methods."""

    @staticmethod
    def read_dataset(dataset_path):
        df = pd.read_csv(dataset_path).fillna(0)
        attrs = [re.sub(r'\W+', '*', a)
                 for a in [col for col in df.columns]]
        return attrs, np.array(df)

    @staticmethod
    def write_result(fn, content):
        dir_path, _ = os.path.split(fn)
        if len(dir_path) > 0 and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(fn, "w") as outfile:
            yaml.dump(content, outfile, default_flow_style=None)
        print('Wrote result to', fn, '\n')

    @staticmethod
    def sdiv(num: float, denom: float, fault: Any = '', mult=True):
        """`safe` division"""
        return fault if denom == 0 else (
                (100 if mult else 1) * num / denom)

    @staticmethod
    def log(label: str, value: Any):
        print(f'{label} '.ljust(18, '-') + str(value))

    @staticmethod
    def logr(label: str, num: float, den: float):
        a, b, ratio = round(num, 0), round(den, 0), \
                      Utility.sdiv(num, den)
        return Utility.log(label, f'{a} of {b} - {ratio:.1f} %')

    @staticmethod
    def logrd(label: str, num: float, den: float):
        Utility.logr(label, num / 100, den / 100)

    @staticmethod
    def normalize(data: np.ndarray, attr_ranges=None):
        """Make sure data is in range 0.0 - 1.0"""
        np.seterr(divide='ignore', invalid='ignore')
        for i in range(data.shape[1]):
            range_max = attr_ranges[i] \
                if attr_ranges is not None else (data[:, i])
            data[:, i] = (data[:, i]) / range_max
            data[:, i] = np.nan_to_num(data[:, i])
        return data

    @staticmethod
    def parse_pred(config: dict):
        """Parse text value of a constraint predicate.
        FIXME: try not use eval here
        """
        result = {}
        for key, value in config.items():
            if isinstance(value, str):
                result[key] = ((key,), eval(value),)
            elif isinstance(value, Sized) \
                    and len(value) == 2 \
                    and isinstance(value[0], Sized) \
                    and isinstance(value[1], str):
                ids, pred = value
                result[key] = (tuple([key] + ids), eval(pred))
            else:
                warnings.warn(f'Invalid constraint format: {value}')
        return result

    @staticmethod
    def time_sec(start: time, end: time) -> int:
        """Time difference between start and end time, in seconds."""
        return round((end - start) / 1e9, 1)
