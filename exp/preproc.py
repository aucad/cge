from typing import List

from exp import CONSTR_DICT
from exp.utility import read_dataset, first_available


def categorize(cd: CONSTR_DICT):
    immutable = [k for k, (_, P) in cd.items() if P is False]
    mutable = dict([x for x in cd.items() if x[0] not in immutable])
    return immutable, mutable


def fmt(text, wchar, *attrs):
    param, fmt_str = (wchar, lambda v: f'{wchar}[{attrs.index(v)}]')
    for t in sorted(attrs, key=len, reverse=True):
        text = text.replace(t, fmt_str(t))
    return f'lambda {param}: {text}'.strip()


def pred_convert(imm: List[str], pred: List[str], attr: List[str]):
    imm = [(k, ((k,), False)) for k in map(attr.index, imm or [])]
    fd, mut, wchar = {}, {}, first_available(attr)
    for p in (pred or []):
        s = [t for t in attr if t in p]
        a = first_available(list(fd.keys()))
        v = fmt(p, wchar, *s)
        sources = list(map(attr.index, s))
        mut[a] = (tuple(sources), eval(v),)
        fd[a] = {'attrs': dict(zip(sources, s)), 'exec': v, 'text': p}
    return {**dict(imm), **dict(mut)}, fd


def pred_parse(config):
    ck, attrs = 'constraints', read_dataset(config['dataset'])[0]
    config[ck], config['p_config'] = pred_convert(
        config[ck]['immutable'], config[ck]['predicates'], attrs)
    return config
