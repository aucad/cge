from os import path
from statistics import mean

import pandas as pd
from pytablewriter import SpaceAlignedTableWriter

from plot import ResultData
from exp import ensure_dir


class TablePlot(ResultData):

    @staticmethod
    def fmt(r):
        ac = r['folds']['accuracy']
        ev = r['folds']['n_evasions']
        vd = r['folds']['n_valid_evades']
        return [
            TablePlot.r_cls(r),
            TablePlot.r_name(r),
            TablePlot.fmt_attack_name(r),
            f"{TablePlot.arr_mean(ac):.1f}",
            f"{TablePlot.fold_mean(ev, r):.1f}",
            f"{TablePlot.fold_mean(vd, r):.1f}",
            round(r['experiment']['duration_sec'], 0)]

    def write_table(self, sorter, out_dir):
        fn = self.fn_pattern('txt', 'plot', out_dir)
        writer = SpaceAlignedTableWriter()
        writer.headers = ('classifier,exp-name,attack,accuracy,'
                          'evades,valid,dur(s)'.split(','))
        mat = [TablePlot.fmt(r) for r in self.raw_rata]
        mat = sorted(mat, key=sorter)
        writer.value_matrix = mat
        print('Results for directory:', self.directory)
        ensure_dir(out_dir)
        writer.write_table()
        writer.dump(fn)

    def show_duration(self):
        div = 72 * "="
        ts = pd.to_timedelta(sum(
            [r['experiment']['end'] - r['experiment']['start']
             for r in self.raw_rata]))
        print(f'{div}\nTotal duration: {ts}\n{div}')


def plot_results(directory, out=None):
    res = TablePlot(directory)
    if res.n_results > 0:
        res.write_table(
            sorter=lambda x: (x[0], x[1], x[2]),
            out_dir=out or directory)
        res.show_duration()
