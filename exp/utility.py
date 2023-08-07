import os
import time
import yaml

from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from networkx import draw_networkx, shell_layout


def attr_of(o, t):
    return [x for x in dir(o) if isinstance(getattr(o, x), t)]


def fname(c):
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


def plot_graph(v, c, a):
    """Plot a constraint-dependency graph."""
    gn = sorted(v.dep_graph.nodes)
    if len(gn) > 0:
        fn = os.path.join(c.out, f'{c.name}_graph.pdf')
        color_map = [
            '#CFD8DC' if n in v.immutable else
            '#00E676' if n not in v.constraints.keys() else
            '#00BCD4' if n in v.single_feat.keys() else
            '#FFC107' for n in gn]
        ensure_dir(fn)
        draw_networkx(
            v.dep_graph,
            pos=shell_layout(v.dep_graph),
            with_labels=True, node_color=color_map, arrowstyle='->',
            font_size=8, font_weight='bold')
        plt.legend(
            labels=[f'{k}: {a[k]}' for k in gn], loc='upper left',
            handles={Patch(fill=False, alpha=0) for _ in gn},
            bbox_to_anchor=(.92, 1), frameon=False)
        plt.savefig(fn, bbox_inches="tight")
