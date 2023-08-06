import time
from collections import namedtuple

import numpy as np
from sklearn.model_selection import KFold

from exp import Utility as Util, Result, Validation, \
    AttackRunner as Attack, ModelTraining as Model, Loggable


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
        self.attr_ranges = {}
        self.attrs = []
        self.start = 0
        self.end = 0

    def conf(self, key: str):
        """Try get configration key."""
        if key and hasattr(self.config, key):
            return getattr(self.config, key)

    def run(self):
        """Run an experiment of K folds."""
        c = self.config

        # load dataset, experiment setup
        self.attrs, rows = Util.read_dataset(c.dataset)
        self.X, self.y = rows[:, :-1], rows[:, -1].astype(int).flatten()
        self.folds = [x for x in KFold(
            n_splits=c.folds, shuffle=True).split(self.X)]
        for col_i in range(len(self.attrs) - 1):
            self.attr_ranges[col_i] = max(self.X[:, col_i])
        self.X = Util.normalize(self.X, self.attr_ranges)
        self.cls = Model(self.conf('xgb'))
        self.attack = Attack(c.iter, c.validate, self.conf('zoo'))
        self.validation = Validation(c.immutable, c.constraints)

        # run K folds
        self.log()
        self.start = time.time_ns()
        for indices in self.folds:
            data = self.X.copy(), self.y.copy()
            self.cls.reset(*data, *indices).train()
            self.result.append(self.cls.score)
            self.attack.reset(self.cls).run(self.validation)
            self.result.append(self.attack.score)
            self.cls.score.log()
            self.attack.score.log()
            print('=' * 52)
        self.end = time.time_ns()
        self.result.log()

        # write results to file
        Util.plot_graph(self.validation.dep_graph, c, self.attrs)
        Util.write_result(Util.dyn_fname(c), self.to_dict())

    def log(self):
        Util.log('Dataset', self.config.dataset)
        Util.log('Record count', len(self.X))
        Util.log('Classifier', self.cls.name)
        Util.log('Classes', len(np.unique(self.y)))
        Util.log('Attributes', len(self.attrs))
        Util.log('Immutable', len(self.validation.immutable))
        Util.log('Constraints', len(self.validation.constraints.keys()))
        Util.log('Attack', self.attack.name)
        Util.log('Attack max iter', self.attack.max_iter)
        Util.log('Validation', self.config.validate)
        Util.log('K-folds', self.config.folds)

    def to_dict(self) -> dict:
        return {'config': {
            'dataset': self.config.dataset,
            'config_path': self.config.config_path,
            'classifier': self.cls.name,
            'attack': self.attack.name,
            'n_records': len(self.X),
            'n_classes': len(np.unique(self.y)),
            'n_class_records': dict(
                [(int(k), int(v)) for k, v in
                 zip(*np.unique(self.y, return_counts=True))]),
            'n_attributes': len(self.attrs),
            'attrs': dict(enumerate(self.attrs)),
            'attr_ranges': dict(
                [(str(k), float(v)) for k, v in
                 zip(self.attrs, self.attr_ranges.values())]),
            'k_folds': self.config.folds,
            'n_attack_max_iter': self.attack.max_iter,
            'duration_sec': Util.time_sec(self.start, self.end),
            'immutable': self.validation.immutable,
            'constraints': list(self.validation.constraints.keys()),
            'predicates': self.conf('str_constraints'),
            'dep_graph': dict(
                [(k, list(v)) for k, v in
                 self.validation.desc.items()]),
        }, 'folds': {**self.result.to_dict()}}
