from os import path
from statistics import mean, stdev

from pytablewriter import SpaceAlignedTableWriter

from plot import ResultData
from exp import ensure_dir


class TablePlot(ResultData):

    def __init__(self, directory, out_dir):
        super().__init__(directory)
        self.out = out_dir

    def ulist(self, fc):
        return list(set([fc(r) for r in self.raw_rata]))

    def make_table(self, fname, headers, mat, sorter=None):
        if self.n_results:
            ensure_dir(self.out)
            fn = self.fn_pattern('txt', fname, self.out)
            srt = sorter or (lambda x: (x[0], x[1], x[2]))
            writer = SpaceAlignedTableWriter()
            writer.headers = headers.split(',')
            writer.value_matrix = sorted(mat, key=srt)
            writer.write_table()
            writer.dump(fn)
        return self

    def evasion_plot(self):
        def fmt(r):
            return [
                self.cls(r),
                self.name(r),
                self.attack(r),
                f"{self.acc(r):.1f}",
                f"{self.evades(r):.1f}",
                f"{self.valid(r):.1f}",
                f"{r['experiment']['duration_sec']:.0f}"]

        hdr = 'classifier,exp-name,attack,' \
              'accuracy,evades,valid,dur(s)'
        mat = [fmt(r) for r in self.raw_rata]
        return self.make_table('plot', hdr, mat)

    def accuracy_plot(self):
        def fmt(m, n):
            recs = [r for r in self.raw_rata if
                    self.cls(r) == m and self.name(r) == n]
            a = mean([self.acc(r) for r in recs])
            s = stdev([self.acc(r) for r in recs])
            return m, n, f"{a:.1f} ± {s:.1f}"

        mat = [i for x in [[
            fmt(m, n) for n in self.ulist(self.name)]
            for m in self.ulist(self.cls)] for i in x]
        hdr = 'classifier,exp-name,accuracy±σ'
        return self.make_table('cls_acc', hdr, mat)

    def config_plot(self):
        def fmt(r):
            e, v = r['experiment'], r['validation']
            i, p = set(v['immutable']), v['predicates']
            d = set([i for x in v['dependencies'].values() for i in x])
            a = set(list(r['experiment']['attrs'])[:-1])
            f = (sum([len(m['attrs']) for m in p.values()]) /
                 max(1, len(p)))
            return tuple([
                self.name(r), e['n_records'], e['n_classes'],
                e['n_attributes'] - 1,  # exclude class label
                len(a - i - d),  # feat that can take on any value
                v['n_constraints'], len(i), len(p), f"{f:.1f}"])

        mat = self.ulist(fmt)
        hdr = 'exp-name,records,cls,ft,*val,#C,⊥,P,ft/P'
        return self.make_table('config', hdr, mat)


def plot_results(directory, out=None):
    print('Results for directory:', directory)
    tp = TablePlot(directory, (out or directory)) \
        .config_plot().accuracy_plot().evasion_plot()

    if tp.n_results > 0:
        tp.show_duration()
