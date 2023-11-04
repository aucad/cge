import sys
import time
from collections import namedtuple

import numpy as np
from sklearn.model_selection import KFold

from exp import Result, Validation, AttackRunner, ClsPicker, \
    score_valid, machine_details
from exp.utility import log, time_sec, write_yaml, file_name, \
    read_dataset


class Experiment:
    """Run adversarial experiment"""

    def __init__(self, conf: namedtuple):
        self.X = None
        self.y = None
        self.cls = None
        self.folds = None
        self.attack = None
        self.validation = None
        self.result = Result()
        self.attr_range = []
        self.attrs = []
        self.start = 0
        self.end = 0
        self.inv_idx = None
        self.config = conf

    def conf(self, key: str):
        """Try get configration key."""
        if key and hasattr(self.config, key):
            return getattr(self.config, key)

    def run(self):
        """Run an experiment of K folds."""
        c = self.config
        self.prepare_input_data()
        self.cls = ClsPicker.load(c.cls)(self.conf(c.cls))
        self.attack = AttackRunner(
            c.attack, c.validate, self.conf(c.attack))
        self.validation = Validation(
            c.constraints, self.attr_range, c.reset_strategy)
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
        write_yaml(file_name(c), self.to_dict())
        return self

    def prepare_input_data(self):
        np.seterr(divide='ignore', invalid='ignore')
        self.attrs, rows = read_dataset(self.config.dataset)
        self.X = rows[:, :-1]
        self.y = rows[:, -1].astype(int).flatten()
        attr_min = np.ones(self.X.shape[1])
        attr_max = np.ones(self.X.shape[1])
        for i in range(self.X.shape[1]):
            attr_min[i] = mn = np.floor(min(self.X[:, i]))
            attr_max[i] = mx = np.ceil(max(self.X[:, i]))
            self.X[:, i] = np.nan_to_num(
                (self.X[:, i] - mn) / (mx - mn))
        self.attr_range = list(zip(attr_min, attr_max))
        self.inv_idx = score_valid(
            self.X, self.X, self.config.constraints, self.attr_range)[1]
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
        log('Classifier', self.cls.name)
        log('Attack', self.attack.name)
        log('Records', len(self.X))
        log('Classes', len(np.unique(self.y)))
        log('Attributes', len(self.attrs))
        log('Constraints', len(self.validation.constraints.keys()))
        log('Immutable', len(self.validation.immutable))
        log('Attack max iter',
            (self.attack.conf['max_iter']
             if 'max_iter' in self.attack.conf else 'N/A'))
        log('Validation', self.config.validate)
        log('K-folds', self.config.folds)

    # noinspection PyTypeChecker
    def to_dict(self) -> dict:
        return {'experiment': {
            'name': self.config.name,
            'dataset': self.config.dataset,
            'description': self.config.desc,
            'config': self.config.config_path,
            'k_folds': self.config.folds,
            'n_records': len(self.X),
            'n_classes': len(np.unique(self.y)),
            'n_attributes': len(self.attrs),
            'attrs': dict(enumerate(self.attrs)),
            'attrs_ranges': dict(
                [(self.attrs[i], [int(mn), int(mx)])
                 for i, (mn, mx) in enumerate(self.attr_range)]),
            'class_distribution': dict(
                [map(int, x) for x in
                 zip(*np.unique(self.y, return_counts=True))]),
            'system': machine_details(),
            'capture_utc': time.time_ns(),
            'duration_sec': time_sec(self.start, self.end),
            'start': self.start, 'end': self.end,
        }, 'validation': {
            'enabled': self.config.validate,
            'n_constraints': len(self.validation.constraints),
            'immutable': self.validation.immutable,
            'predicates': self.conf('p_config'),
            'dependencies': dict(self.validation.deps.items()),
            'reset_strategy': self.config.reset_strategy
        }, 'invalid_rows': self.inv_idx.tolist(),
            'classifier': {**self.cls.to_dict()},
            'attack': {**self.attack.to_dict()},
            'folds': {**self.result.to_dict()}}
