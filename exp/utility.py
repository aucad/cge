import os
import re
import time
from collections import namedtuple
from itertools import product
from typing import List

import numpy as np
import pandas as pd
import yaml


def to_namedtuple(d: dict):
    return namedtuple('exp', (",".join(list(d.keys()))))(**d)


def first_available(taken: List[str], init: List[str] = None):
    """first lowercase char sequence that doesn't occur in taken"""
    cmap = [c for c in list(map(chr, range(ord('a'), ord('z') + 1)))]
    cmap = [f'{c}{d}' for c, d in product(cmap, init)] if init else cmap
    first = [c for c in cmap if c not in taken]
    return first[0] if first else first_available(taken, cmap)


def attr_fmt(attr: List[str]):
    return [re.sub(r'^\w=_-', '*', col) for col in attr]


def read_dataset(dataset_path):
    df = pd.read_csv(dataset_path).fillna(0)
    return attr_fmt(df.columns), np.array(df)


def attr_of(o, t):
    return [x for x in dir(o) if isinstance(getattr(o, x), t)]


def upper_attrs(cname):
    upper_str = [x for x in attr_of(cname, str) if x.isupper()]
    return sorted([getattr(cname, x) for x in upper_str])


def file_name(c):
    r = str(round(time.time() * 1000))[-3:]
    v = "" if c.validate else "REG_"
    a, s, n = c.attack, c.cls, c.name
    i = getattr(c, a)['max_iter'] if 'max_iter' in getattr(c, a) else 'auto'
    return os.path.join(c.out, f'{v}{n}_{s}_{a}_{i}_{r}.yaml')


def ensure_dir(fpath):
    dir_path, _ = os.path.split(fpath)
    if len(dir_path) > 0 and not os.path.exists(dir_path):
        os.makedirs(dir_path)


def write_yaml(fn, content):
    ensure_dir(fn)
    with open(fn, "w") as outfile:
        yaml.dump(content, outfile, default_flow_style=None)
    print('Wrote result to', fn, '\n')


def sdiv(n: float, d: float, fault='', mult=True):
    return fault if d == 0 else (100 if mult else 1) * n / d


def log(label: str, value):
    print(f'{label} '.ljust(18, '-') + ' ' + str(value))


def logr(label: str, n: float, d: float):
    a, b, r = round(n, 0), round(d, 0), sdiv(n, d)
    return log(label, f'{a} of {b} - {r:.2f} %')


def logd(label: str, n: float, d: float):
    logr(label, n / 100, d / 100)


def time_sec(start: time, end: time) -> int:
    """Time difference in seconds."""
    return round((end - start) / 1e9, 1)
