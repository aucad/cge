import os
import time
import yaml
import re
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from networkx import draw_networkx, shell_layout


def attr_fmt(attr: List[str]):
    return [re.sub(r'^\w=_-', '*', col) for col in attr]


def read_dataset(dataset_path):
    df = pd.read_csv(dataset_path).fillna(0)
    return attr_fmt(df.columns), np.array(df)


def attr_of(o, t):
    return [x for x in dir(o) if isinstance(getattr(o, x), t)]


def fname(c):
    r = str(round(time.time() * 1000))[-4:]
    v = "" if c.validate else "_F"
    return os.path.join(c.out, f'{c.name}{v}_{r}.yaml')


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
        fn = os.path.join(c.out, f'__graph_{c.name}.pdf')
        meaning = ['any-value', 'immutable', 'single-feat', 'multi-feat']
        colors = ['#00E676', '#CFD8DC', '#00BCD4', '#FFC107']
        color_map = [
            colors[1] if n in v.immutable else
            colors[0] if n not in v.constraints.keys() else
            colors[2] if n in v.single_feat.keys() else
            colors[3] for n in gn]
        leg_colors = [i for i in sorted(list(set(
            [colors.index(c) for c in color_map])))]
        ensure_dir(fn)
        ax = plt.figure(1).add_subplot(1, 1, 1)
        draw_networkx(
            v.dep_graph, pos=shell_layout(v.dep_graph),
            with_labels=True, node_color=color_map, arrowstyle='->',
            font_size=8, font_weight='bold', ax=ax)
        legend1 = plt.legend(
            labels=[f'{k}: {a[k]}' for k in gn], loc='upper left',
            handles={Patch(fill=False, alpha=0) for _ in gn},
            bbox_to_anchor=(.92, 1.02), frameon=False)
        legend2 = plt.legend(
            frameon=False, borderpad=0, handlelength=1, handleheight=1,
            ncol=len(leg_colors), loc='lower left', bbox_to_anchor=(0, -0.09),
            handles={Patch(fill=True, color=colors[i]) for i in leg_colors},
            labels=[meaning[i] for i in leg_colors])
        ax.add_artist(legend1)
        ax.add_artist(legend2)
        plt.savefig(fn, bbox_inches="tight")
