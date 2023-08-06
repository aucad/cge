import sys
import time
from collections import namedtuple

import numpy as np
from sklearn.model_selection import KFold

from exp import Result, Validation, AttackRunner, ModelTraining
from exp.utility import read_dataset, log, plot_graph, time_sec, \
    write_result, dyn_fname
from exp.types import Loggable


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
        self.attrs, rows = read_dataset(c.dataset)
        self.X = rows[:, :-1]
        self.y = rows[:, -1].astype(int).flatten()
        self.folds = [x for x in KFold(
            n_splits=c.folds, shuffle=True).split(self.X)]
        self.cls = ModelTraining(self.conf('xgb'))
        self.attack = AttackRunner(c.iter, c.validate, self.conf('zoo'))
        self.validation = Validation(c.immutable, c.constraints)

        # run K folds
        self.log()
        self.start = time.time_ns()
        for fold_num, indices in enumerate(self.folds):
            print('-' * 50)
            data = self.X.copy(), self.y.copy()
            self.cls.reset(*data, *indices).train()
            self.result.append(self.cls.score)
            self.attack.reset(self.cls).run(self.validation)
            self.result.append(self.attack.score)
            sys.stdout.write('\x1b[1A')
            sys.stdout.write('\x1b[2K')
            log('Fold #', fold_num + 1)
            self.cls.score.log()
            self.attack.score.log()
        self.end = time.time_ns()
        self.result.log()

        # write results to file
        plot_graph(self.validation.dep_graph, c, self.attrs)
        write_result(dyn_fname(c), self.to_dict())

    def log(self):
        log('Dataset', self.config.dataset)
        log('Record count', len(self.X))
        log('Classifier', self.cls.name)
        log('Classes', len(np.unique(self.y)))
        log('Attributes', len(self.attrs))
        log('Immutable', len(self.validation.immutable))
        log('Constraints', len(self.validation.constraints.keys()))
        log('Attack', self.attack.name)
        log('Attack max iter', self.attack.max_iter)
        log('Validation', self.config.validate)
        log('K-folds', self.config.folds)

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
            'k_folds': self.config.folds,
            'n_attack_max_iter': self.attack.max_iter,
            'duration_sec': time_sec(self.start, self.end),
            'immutable': self.validation.immutable,
            'constraints': list(self.validation.constraints.keys()),
            'predicates': self.conf('str_constraints'),
            'dep_graph': dict(
                [(k, list(v)) for k, v in
                 self.validation.desc.items()]),
        }, 'folds': {**self.result.to_dict()}}
