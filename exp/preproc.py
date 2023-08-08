from typing import Dict, List

from exp import CONSTR_DICT
from exp.utility import read_dataset


def categorize(cd: CONSTR_DICT):
    immutable = [k for k, (_, P) in cd.items() if P is False]
    single_ft = dict([
        (t, P) for (t, (s, P)) in cd.items() if (t,) == s and P])
    multi_ft = dict([
        x for x in cd.items()
        if x[0] not in single_ft and x[0] not in immutable])
    return immutable, single_ft, multi_ft


def fmt(text, *attrs):
    option_1 = ('x', lambda v: 'x')
    option_2 = ('a', lambda v: f'a[{attrs.index(v)}]')
    param, fmt_str = option_1 if len(attrs) == 1 else option_2
    for t in sorted(attrs, key=len, reverse=True):
        text = text.replace(t, fmt_str(t))
    return f'lambda {param}: {text}'.strip()


def parse_pred(conf) -> CONSTR_DICT:
    single = [(k, ((k,), eval(v),))
              for k, v in conf.items() if isinstance(v, str)]
    multi = [(k, (tuple([k] + v[0]), eval(v[1]),))
             for k, v in conf.items() if len(v) == 2]
    return {**dict(single), **dict(multi)}


def pred_convert(items: Dict[str, str], attr: List[str]):
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


def pred_parse(config):
    c_key = 'constraints'
    if c_key in config.keys():
        attrs, _ = read_dataset(config['dataset'])
        config['str_' + c_key] = dict(
            [(str(k), str(v).strip()) for (k, v) in
             config[c_key].items()])
        if config[c_key] is not None:
            config[c_key], str_c = \
                pred_convert(config[c_key], attrs)
        else:
            config[c_key], str_c = {}, {}
        config['str_func'] = dict(
            [(k, v if isinstance(v, str) else list(v))
             for k, v in str_c.items()])
    return config
