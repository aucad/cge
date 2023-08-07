import os
import re
import time
from typing import Tuple

import yaml

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from networkx import draw_networkx, spring_layout

from exp.types import CONSTR_DICT, CONFIG_CONST_DICT


def attr_of(o, t):
    return [x for x in dir(o) if isinstance(getattr(o, x), t)]


def read_dataset(dataset_path):
    df = pd.read_csv(dataset_path).fillna(0)
    attrs = [re.sub(r'^\w=_-', '*', col) for col in df.columns]
    return attrs, np.array(df)


def dyn_fname(c):
    v = "" if c.validate else "_F"
    return os.path.join(c.out, f'{c.name}_i{c.iter}{v}.yaml')


def ensure_dir(fpath):
    dir_path, _ = os.path.split(fpath)
    if len(dir_path) > 0 and not os.path.exists(dir_path):
        os.makedirs(dir_path)


def write_result(fn, content):
    ensure_dir(fn)
    with open(fn, "w") as outfile:
        yaml.dump(content, outfile, default_flow_style=None)
    print('Wrote result to', fn, '\n')


def normalize(data: np.ndarray, attr_ranges=None):
    """Make sure data is in range 0.0 - 1.0"""
    np.seterr(divide='ignore', invalid='ignore')
    for i in range(data.shape[1]):
        range_max = attr_ranges[i] \
            if attr_ranges is not None else (data[:, i])
        data[:, i] = (data[:, i]) / range_max
        data[:, i] = np.nan_to_num(data[:, i])
    return data


def sdiv(n: float, d: float, fault='', mult=True):
    return fault if d == 0 else (100 if mult else 1) * n / d


def log(label: str, value):
    print(f'{label} '.ljust(18, '-') + ' ' + str(value))


def logr(label: str, n: float, d: float):
    a, b, r = round(n, 0), round(d, 0), sdiv(n, d)
    return log(label, f'{a} of {b} - {r:.1f} %')


def logrd(label: str, n: float, d: float):
    logr(label, n / 100, d / 100)


def time_sec(start: time, end: time) -> int:
    """Time difference in seconds."""
    return round((end - start) / 1e9, 1)


def parse_pred(conf: CONFIG_CONST_DICT) -> CONSTR_DICT:
    """Parse constraints from a text config file."""
    single = [(k, ((k,), eval(v),))
              for k, v in conf.items() if isinstance(v, str)]
    multi = [(k, (tuple([k] + v[0]), eval(v[1]),))
             for k, v in conf.items() if len(v) == 2]
    return {**dict(single), **dict(multi)}


def pred_convert(items: dict[str, str], attr) \
        -> Tuple[CONFIG_CONST_DICT, CONFIG_CONST_DICT]:
    result, f_dict = {}, {}
    for a in [a for a in attr if a in items.keys()]:
        k, txt = attr.index(a), items[a]
        if txt is False or bool(txt) is False:
            result[k] = ((k,), False)
            continue
        s = [a] + [t for t in attr if t != a and t in txt]
        idx = tuple([attr.index(x) for x in s])
        if len(s) == 1:
            f_text = 'lambda x: ' + txt.replace(a, 'x')
        else:
            for t in sorted(s, key=len, reverse=True):
                txt = txt.replace(t, f'a[{s.index(t)}]')
            f_text = 'lambda a: ' + txt
        f_dict[k] = (idx, f_text)
        result[k] = idx, eval(f_text)
    return result, f_dict


def plot_graph(v, c, a):
    """Plot a constraint-dependency graph."""
    if len(gn := sorted(v.dep_graph.nodes)) > 0:
        fn = os.path.join(c.out, f'{c.name}_graph.pdf')
        color_map = [
            '#CFD8DC' if n in v.immutable else
            '#00E676' if n not in v.constraints.keys() else
            '#00BCD4' if n in v.single_feat.keys() else
            '#FFC107' for n in gn]
        ensure_dir(fn)
        draw_networkx(
            v.dep_graph, spring_layout(v.dep_graph),
            with_labels=True, node_color=color_map, arrowstyle='->',
            font_size=8, font_weight='bold')
        plt.legend(
            labels=[f'{k}: {a[k]}' for k in gn], loc='upper left',
            handles={Patch(fill=False, alpha=0) for _ in gn},
            bbox_to_anchor=(.92, 1), frameon=False)
        plt.savefig(fn, bbox_inches="tight")
