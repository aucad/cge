import json
import re
from typing import Any

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
        with open(fn, "w") as outfile:
            json.dump(content, outfile, indent=4)
        print('Wrote result to', fn, '\n')

    @staticmethod
    def sdiv(num, denom, fault: Any = '', mult=True):
        return fault if denom == 0 else (
                (100 if mult else 1) * num / denom)

    @staticmethod
    def log(label: str, value):
        print(f'{label} '.ljust(18, '-') + str(value))

    @staticmethod
    def logr(label, num, den):
        a, b, ratio = round(num, 0), round(den, 0), \
                      Utility.sdiv(num, den)
        return Utility.log(label, f'{a} of {b} - {ratio:.1f} %')

    @staticmethod
    def logrd(label, num, den):
        Utility.logr(label, num / 100, den / 100)

    @staticmethod
    def normalize(data, attr_ranges=None):
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
        FIXME: find some better approach that does not use eval.
        """
        result = {}
        for key, value in config.items():
            result[key] = eval(value)
        return result
