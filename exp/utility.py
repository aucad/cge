import os
import re
import time
import warnings
import yaml
from typing import Any, Sized

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from networkx import draw_networkx, spring_layout

from exp import CONSTR_DICT, CONFIG_CONST_DICT


class Utility:
    """Miscellaneous helper methods."""

    @staticmethod
    def read_dataset(dataset_path):
        df = pd.read_csv(dataset_path).fillna(0)
        attrs = [re.sub(r'\W+', '*', a)
                 for a in [col for col in df.columns]]
        return attrs, np.array(df)

    @staticmethod
    def dyn_fname(c):
        """Generate filename for where to save experiment results."""
        v = "T" if c.validate else "F"
        return os.path.join(c.out, f'{c.name}_i{c.iter}_{v}.yaml')

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
    def time_sec(start: time, end: time) -> int:
        """Time difference between start and end time, in seconds."""
        return round((end - start) / 1e9, 1)

    @staticmethod
    def parse_pred(config: CONFIG_CONST_DICT) -> CONSTR_DICT:
        """Parse constraints from yaml text config file."""
        result = {}
        for key, value in config.items():
            if isinstance(value, str):
                result[key] = ((key,), eval(value),)
            elif isinstance(value, Sized) and len(value) == 2 \
                    and isinstance(value[0], Sized) \
                    and isinstance(value[1], str):
                ids, pred = value
                result[key] = (tuple([key] + ids), eval(pred))
            else:
                warnings.warn(f'Invalid constraint format: {value}')
        return result

    @staticmethod
    def plot_graph(dep_graph, config, attrs):
        """Plot constraint dependency graph."""
        fn = os.path.join(config.out, f'{config.name}_graph.pdf')
        node_names = dict([(i, n) for i, n in enumerate(attrs)])
        if dep_graph:
            fig = plt.figure()
            draw_networkx(
                dep_graph, ax=fig.add_subplot(),
                pos=spring_layout(dep_graph, k=0.25),
                node_color='orange', with_labels=True)
            plt.legend(
                handles={patches.Patch(
                    fill=False, alpha=0, label=k)
                    for k in dep_graph.nodes},
                labels=[f'{k} : {node_names[k]}'
                        for k in sorted(dep_graph.nodes)],
                handletextpad=-0.25, loc='upper left',
                bbox_to_anchor=(.95, 1), frameon=False)
            plt.savefig(fn, bbox_inches="tight")
            plt.clf()
