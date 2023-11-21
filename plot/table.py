from os import path
from statistics import mean, stdev

from pytablewriter import SpaceAlignedTableWriter

from plot import ResultData
from exp import ensure_dir


class TablePlot(ResultData):

    def __init__(self, directory, out_dir):
        super().__init__(directory)
        self.out = out_dir

    def make_table(self, fn, mat):
        if self.n_results == 0:
            return
        ensure_dir(self.out)
        writer = SpaceAlignedTableWriter()
        writer.headers = self.headers
        if self.sorter:
            mat = sorted(mat, key=self.sorter)
        writer.value_matrix = mat
        writer.write_table()
        writer.dump(fn)

    @property
    def headers(self):
        return ""

    @property
    def sorter(self):
        return lambda x: (x[0], x[1], x[2])


class EvasionPlot(TablePlot):

    @property
    def headers(self):
        return ('classifier,exp-name,attack,accuracy,' +
                'evades,valid,dur(s)').split(',')

    @property
    def sorter(self):
        return lambda x: (x[0], x[1], x[2])

    @staticmethod
    def fmt(r):
        return [
            EvasionPlot.cls(r),
            EvasionPlot.name(r),
            EvasionPlot.attack(r),
            f"{EvasionPlot.acc(r):.1f}",
            f"{EvasionPlot.evades(r):.1f}",
            f"{EvasionPlot.valid(r):.1f}",
            round(r['experiment']['duration_sec'], 0)]

    def write_table(self):
        fn = self.fn_pattern('txt', 'plot', self.out)
        mat = [EvasionPlot.fmt(r) for r in self.raw_rata]
        self.make_table(fn, mat)
        return self


class AccuracyPlot(TablePlot):

    @property
    def headers(self):
        return 'classifier,exp-name,accuracy ± std'.split(',')

    @property
    def sorter(self):
        return lambda x: (x[0], x[1])

    def calc_acc(self, model, name):
        recs = [r for r in self.raw_rata if
                AccuracyPlot.cls(r) == model and
                AccuracyPlot.name(r) == name]
        avg = mean([AccuracyPlot.acc(r) for r in recs])
        std = stdev([AccuracyPlot.acc(r) for r in recs])
        return avg, std

    @staticmethod
    def num_fmt(a, b, sep='±'):
        return f"{a:.1f} {sep} {b:.1f}"

    def write_table(self):
        fn, mat = self.fn_pattern('txt', 'cls_acc', self.out), []
        models = [AccuracyPlot.cls(r) for r in self.raw_rata]
        names = [AccuracyPlot.name(r) for r in self.raw_rata]
        for m in list(set(models)):
            for n in list(set(names)):
                acc = self.calc_acc(m, n)
                mat.append([m, n, self.num_fmt(*acc)])
        self.make_table(fn, mat)


def plot_results(directory, out=None):
    print('Results for directory:', directory)
    AccuracyPlot(directory, (out or directory)).write_table()
    res = EvasionPlot(directory, (out or directory)).write_table()
    if res.n_results > 0:
        res.show_duration()
