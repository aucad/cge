from typing import Dict, List

from exp import CONSTR_DICT
from exp.utility import read_dataset


def categorize(cd: CONSTR_DICT):
    immutable = [k for k, (_, P) in cd.items() if P is False]
    mutable = dict([x for x in cd.items() if x[0] not in immutable])
    return immutable, mutable


def find_free_var(attrs):
    available_lowercase = \
        [c for c in list(map(chr, range(ord('a'), ord('z') + 1)))
         if c not in attrs]
    if not available_lowercase:
        raise ValueError('attribute conflict; try rename attributes')
    return available_lowercase[0]


def fmt(text, *attrs):
    wchar = find_free_var(attrs)
    param, fmt_str = (wchar, lambda v: f'{wchar}[{attrs.index(v)}]')
    for t in sorted(attrs, key=len, reverse=True):
        text = text.replace(t, fmt_str(t))
    return f'lambda {param}: {text}'.strip()


def pred_convert(items: Dict[str, str], attr: List[str]):
    both, f_dict = [a for a in attr if a in items.keys()], {}
    imm = dict([(k, ((k,), False)) for k in [
        attr.index(a) for a in both
        if items[a] is False or bool(items[a]) is False]])
    for a in [a for a in both if attr.index(a) not in imm]:
        s = [a] + [t for t in attr if t != a and t in items[a]]
        value = fmt(items[a], *s)
        sources = [attr.index(x) for x in s]
        f_dict[attr.index(a)] = (sources, value)
    mut = [(k, (tuple(s), eval(v),)) for k, (s, v) in f_dict.items()]
    result = {**imm, **dict(mut)}
    return result, [[list(s), v] for s, v in f_dict.values()]


def pred_parse(config):
    ck = 'constraints'
    if ck in config.keys() and config[ck]:
        (attrs, _), c = read_dataset(config['dataset']), config[ck]
        config[f'str_{ck}'] = dict([
            (str(k), str(v).strip(),) for k, v in c.items()])
        config[ck], str_c = pred_convert(c, attrs)
        config['p_config'] = str_c
    return config
