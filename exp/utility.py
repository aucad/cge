import os
import time
import yaml
import re
from typing import List

import numpy as np
import pandas as pd


def attr_fmt(attr: List[str]):
    return [re.sub(r'^\w=_-', '*', col) for col in attr]


def read_dataset(dataset_path):
    df = pd.read_csv(dataset_path).fillna(0)
    return attr_fmt(df.columns), np.array(df)


def attr_of(o, t):
    return [x for x in dir(o) if isinstance(getattr(o, x), t)]


def fname(c):
    r = str(round(time.time() * 1000))[-3:]
    v = "" if c.validate else "REG_"
    a = c.attack
    i = getattr(c, a)['max_iter']
    return os.path.join(c.out, f'{v}{c.name}_{a}_{i}_{r}.yaml')


def ensure_dir(fpath):
    dir_path, _ = os.path.split(fpath)
    if len(dir_path) > 0 and not os.path.exists(dir_path):
        os.makedirs(dir_path)


def write_result(fn, content):
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
    return log(label, f'{a} of {b} - {r:.1f} %')


def logrd(label: str, n: float, d: float):
    logr(label, n / 100, d / 100)


def time_sec(start: time, end: time) -> int:
    """Time difference in seconds."""
    return round((end - start) / 1e9, 1)
