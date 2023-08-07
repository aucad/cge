import re
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd

from exp.types import CONSTR_DICT, CONFIG_CONST_DICT


def attr_fmt(attr: List[str]):
    return [re.sub(r'^\w=_-', '*', col) for col in attr]


def read_dataset(dataset_path):
    df = pd.read_csv(dataset_path).fillna(0)
    return attr_fmt(df.columns), np.array(df)


def normalize(data: np.ndarray, attr_ranges=None):
    np.seterr(divide='ignore', invalid='ignore')
    for i in range(data.shape[1]):
        range_max = attr_ranges[i] \
            if attr_ranges is not None else (data[:, i])
        data[:, i] = np.nan_to_num((data[:, i]) / range_max)
    return data


def fmt(text, *attrs):
    option_1 = ('x', lambda v: 'x')
    option_2 = ('a', lambda v: f'a[{attrs.index(v)}]')
    param, fmt_str = option_1 if len(attrs) == 1 else option_2
    for t in sorted(attrs, key=len, reverse=True):
        text = text.replace(t, fmt_str(t))
    return f'lambda {param}: {text}'


def parse_pred(conf: CONFIG_CONST_DICT) -> CONSTR_DICT:
    single = [(k, ((k,), eval(v),))
              for k, v in conf.items() if isinstance(v, str)]
    multi = [(k, (tuple([k] + v[0]), eval(v[1]),))
             for k, v in conf.items() if len(v) == 2]
    return {**dict(single), **dict(multi)}


def pred_convert(items: Dict[str, str], attr: List[str]) \
        -> Tuple[CONFIG_CONST_DICT, CONFIG_CONST_DICT]:
    both, f_dict = [a for a in attr if a in items.keys()], {}
    imm = dict([(k, ((k,), False)) for k in [
        attr.index(a) for a in both
        if items[a] is False or bool(items[a]) is False]])
    for a in [a for a in both if attr.index(a) not in imm]:
        s = [a] + [t for t in attr if t != a and t in items[a]]
        value = fmt(items[a], *s)
        f_dict[attr.index(a)] = value if len(s) == 1 \
            else ([attr.index(x) for x in s[1:]], value)
    result = {**imm, **parse_pred(f_dict)}
    return result, f_dict


def parse_pred_config(config):
    const = 'constraints'
    if const in config.keys():
        attrs, _ = read_dataset(config['dataset'])
        config['str_' + const] = config[const]
        config[const], config['str_func'] = (
            pred_convert(config[const], attrs) if
            config[const] is not None else ({}, {}))
    return config
