import sys
import time
from collections import namedtuple

import numpy as np
from sklearn.model_selection import KFold

from exp import Result, Validation, AttackRunner, ModelTraining, \
    Loggable, score_valid, plot_graph
from exp.utility import log, time_sec, write_result, fname, read_dataset


class Experiment(Loggable):
    """Run adversarial experiment"""

    def __init__(self, conf: dict):
        c_keys = ",".join(list(conf.keys()))
        self.config = (namedtuple('exp', c_keys)(**conf))
        self.X = None
        self.y = None
        self.cls = None
        self.folds = None
        self.attack = None
        self.validation = None
        self.result = Result()
        self.attr_max = np.array([])
        self.attrs = []
        self.start = 0
        self.end = 0
        self.inv_idx = None

    def conf(self, key: str):
        """Try get configration key."""
        if key and hasattr(self.config, key):
            return getattr(self.config, key)

    def run(self):
        """Run an experiment of K folds."""
        c = self.config
        self.prepare_input_data()
        self.cls = ModelTraining(self.conf('xgb'))
        self.attack = AttackRunner(
            c.attack, c.validate, self.conf(c.attack))
        self.validation = Validation(c.constraints, self.attr_max)
        self.log()

        self.start = time.time_ns()
        for fold_num, indices in enumerate(self.folds):
            data = self.X.copy(), self.y.copy()
            self.cls.reset(*data, *indices).train()
            self.result.append(self.cls.score)
            self.attack.reset(self.cls).run(self.validation)
            self.result.append(self.attack.score)
            print('-' * 50)
            log('Fold #', fold_num + 1)
            self.cls.score.log()
            self.attack.score.log()
        self.end = time.time_ns()

        self.result.log()
        plot_graph(self.validation, c, self.attrs)
        write_result(fname(c), self.to_dict())

    def prepare_input_data(self):
        np.seterr(divide='ignore', invalid='ignore')
        self.attrs, rows = read_dataset(self.config.dataset)
        self.X = rows[:, :-1]
        self.y = rows[:, -1].astype(int).flatten()
        self.attr_max = np.ones(self.X.shape[1])
        for i in range(self.X.shape[1]):
            self.attr_max[i] = mx = max(self.X[:, i])
            self.X[:, i] = np.nan_to_num(self.X[:, i] / mx)
        self.inv_idx = score_valid(
            self.X, self.X, self.config.constraints, self.attr_max)[1]
        if len(self.inv_idx) > 0:
            self.X = np.delete(self.X, self.inv_idx, 0)
            self.y = np.delete(self.y, self.inv_idx, 0)
            print(f'WARNING: {len(self.inv_idx)} invalid records.')
            print('They were excluded from input data.')
        if self.X.shape[0] < self.config.folds:
            print('Insufficient input, terminating')
            sys.exit(1)
        self.folds = [x for x in KFold(
            n_splits=self.config.folds, shuffle=True).split(self.X)]

    def log(self):
        log('Dataset', self.config.dataset)
        log('Record count', len(self.X))
        log('Classifier', self.cls.name)
        log('Classes', len(np.unique(self.y)))
        log('Attributes', len(self.attrs))
        log('Constraints', len(self.validation.constraints.keys()))
        log('Immutable', len(self.validation.immutable))
        log('Attack', self.attack.name)
        log('Attack max iter', self.attack.conf['max_iter'])
        log('Validation', self.config.validate)
        log('K-folds', self.config.folds)

    def to_dict(self) -> dict:
        return {'experiment': {
            'dataset': self.config.dataset,
            'description': self.config.desc,
            'config': self.config.config_path,
            'k_folds': self.config.folds,
            'duration_sec': time_sec(self.start, self.end),
        }, 'classifier': {
            'name': self.cls.name,
            'n_records': len(self.X),
            'n_classes': len(np.unique(self.y)),
            'n_attributes': len(self.attrs),
            'attrs': dict(enumerate(self.attrs)),
            'config': self.cls.conf,
            'class_distribution': dict([
                (int(k), int(v)) for k, v in
                zip(*np.unique(self.y, return_counts=True))]),
            'attr_range': dict([
                (k, float(v)) for k, v in enumerate(self.attr_max)]),
        }, 'validation': {
            'original_invalid_rows': self.inv_idx.tolist(),
            'constraints': list(self.validation.constraints.keys()),
            'immutable': self.validation.immutable,
            'configuration': self.conf('str_constraints'),
            'mutable': self.conf('str_func'),
            'dependencies': dict([
                (k, list(v)) for k, v in
                self.validation.desc.items() if len(v) > 0])},
            'attack': {**self.attack.to_dict()},
            'folds': {**self.result.to_dict()}}
