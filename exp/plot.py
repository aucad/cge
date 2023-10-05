from glob import glob
from os import path
from statistics import mean

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from networkx import draw_networkx, shell_layout
from pytablewriter import SpaceAlignedTableWriter

from exp.utility import ensure_dir, read_yaml


def plot_graph(v, c, a):
    """Plot a constraint-dependency graph."""
    gn = sorted(v.graph.nodes)
    if len(gn) > 0:
        fn = path.join(c.out, f'__graph_{c.name}.pdf')
        lbl, clr = ['immutable', 'mutable'], ['#CFD8DC', '#FDD835']
        nc = [clr[0] if n in v.immutable else clr[1] for n in gn]
        ax = plt.figure(1).add_subplot(1, 1, 1)
        draw_networkx(
            v.graph, ax=ax, pos=shell_layout(v.graph),
            with_labels=True, node_color=nc, linewidths=.75,
            width=.75, font_size=8, font_weight='bold')
        legend1 = plt.legend(
            labels=[f'{k:>2}: {a[k]}' for k in gn], loc='upper left',
            handles={Patch(fill=False, alpha=0) for _ in gn},
            ncol=(len(gn) // 20) + 1 if len(gn) > 25 else 1,
            bbox_to_anchor=(.93, 1.02), frameon=False)
        pairs = [(Patch(fill=True, color=clr[i]), lbl[i])
                 for i, _ in enumerate(clr)]
        legend2 = plt.legend(
            *zip(*pairs), frameon=False, borderpad=0,
            handlelength=1, handleheight=1, ncol=1,
            loc='lower left', prop={"size": 8})
        ax.add_artist(legend1)
        ax.add_artist(legend2)
        ensure_dir(fn)
        plt.savefig(fn, bbox_inches="tight")


class ResultData:

    def __init__(self, directory):
        self.raw_rata = []
        self.directory = directory
        for file in glob(path.join(self.directory, "*.yaml")):
            # noinspection PyBroadException
            try:
                self.raw_rata.append(read_yaml(file))
            # flake8: noqa: E722
            except:
                pass

    @property
    def n_results(self):
        return len(self.raw_rata)

    @staticmethod
    def fmt(r):
        def arr_mean(x):
            d = sum(x)
            v = sum(r['folds']['n_records'])
            return f"{100 * (d / v):.1f}"

        return [
            r['classifier']['name'],
            (r['experiment']['name'] if 'name' in r['experiment'] else
             r['experiment']['dataset']),
            r['attack']['name'],
            f"{100 * mean(r['folds']['accuracy']):.1f}",
            arr_mean(r['folds']['n_evasions']),
            arr_mean(r['folds']['n_valid_evades']),
            round(r['experiment']['duration_sec'], 0)]

    def write_table(self, sorter, our_dir):
        flat_name = self.directory.replace('/', '_')
        file_ext, file_name = 'txt', f'__plot_{flat_name}'
        fn = path.join(our_dir, f'{file_name}.{file_ext}')
        writer = SpaceAlignedTableWriter()
        writer.headers = ('classifier,data set,attack,accuracy,'
                          'evades,valid,dur(s)'.split(','))
        mat = [ResultData.fmt(r) for r in self.raw_rata]
        mat = sorted(mat, key=sorter)
        writer.value_matrix = mat
        print('Results for directory:', self.directory)
        writer.write_table()
        writer.dump(fn)

    def show_duration(self):
        div = 72 * "="
        ts = pd.to_timedelta(sum(
            [r['experiment']['end'] - r['experiment']['start']
             for r in self.raw_rata]))
        print(f'{div}\nTotal duration: {ts}\n{div}')


def plot_results(directory, out=None):
    res = ResultData(directory)
    if res.n_results > 0:
        res.write_table(
            sorter=lambda x: (x[0], x[1], x[2]),
            our_dir=out or directory)
        res.show_duration()
