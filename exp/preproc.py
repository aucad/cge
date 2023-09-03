from typing import List

from exp import CONSTR_DICT
from exp.utility import read_dataset


def categorize(cd: CONSTR_DICT):
    immutable = [k for k, (_, P) in cd.items() if P is False]
    mutable = dict([x for x in cd.items() if x[0] not in immutable])
    return immutable, mutable


def first_available(taken):
    cmap = [c for c in list(map(chr, range(ord('a'), ord('z') + 1)))]
    for c_ in [c for c in cmap if c not in taken]:
        return c_
    nested = [[f'{c}{d}' for c in cmap] for d in cmap]
    for cd in [c for lst in nested for c in lst if c not in taken]:
        return cd
    raise ValueError('attribute conflict; try rename attributes')


def fmt(text, *attrs):
    wchar = first_available(attrs)
    param, fmt_str = (wchar, lambda v: f'{wchar}[{attrs.index(v)}]')
    for t in sorted(attrs, key=len, reverse=True):
        text = text.replace(t, fmt_str(t))
    return f'lambda {param}: {text}'.strip()


def pred_convert(imm: List[str], pred: List[str], attr: List[str]):
    imm = [(k, ((k,), False))
           for k in [attr.index(a) for a in imm or []]]
    fd, mut = {}, {}
    for p in (pred or []):
        s = [t for t in attr if t in p]
        value, a = fmt(p, *s), first_available(fd.keys())
        sources = [attr.index(x) for x in s]
        mut[a] = (tuple(sources), eval(value),)
        fd[a] = {'attrs': list(sources), 'exec': value, 'config': p}
    return {**dict(imm), **dict(mut)}, fd


def pred_parse(config):
    ck = 'constraints'
    if ck in config.keys() and config[ck]:
        (attrs, _) = read_dataset(config['dataset'])
        config[ck], config['p_config'] = pred_convert(
            config[ck]['immutable'], config[ck]['predicates'], attrs)
    return config
