from glob import glob
from os import path
from statistics import mean

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from pytablewriter import SpaceAlignedTableWriter

from exp.utility import ensure_dir, read_yaml


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
    def fmt_attack_name(r):
        tmp = r['attack']['name']
        if tmp == 'ProjectedGradientDescent':
            return 'PGD'
        if tmp == 'ZooAttack':
            return 'Zoo'
        if tmp == 'HopSkipJump':
            return 'HSJ'
        if tmp == 'CPGD':
            if not r['attack']['config']['args']['enable_constraints']:
                return 'CPGD[R]'
        return tmp

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
            ResultData.fmt_attack_name(r),
            f"{100 * mean(r['folds']['accuracy']):.1f}",
            arr_mean(r['folds']['n_evasions']),
            arr_mean(r['folds']['n_valid_evades']),
            round(r['experiment']['duration_sec'], 0)]

    def write_table(self, sorter, our_dir):
        flat_name = self.directory.replace('/', '_')
        file_ext, file_name = 'txt', f'__plot_{flat_name}'
        fn = path.join(our_dir, f'{file_name}.{file_ext}')
        writer = SpaceAlignedTableWriter()
        writer.headers = ('classifier,exp-name,attack,accuracy,'
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
