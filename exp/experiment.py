import time
from collections import namedtuple
from os import path
from typing import List

import numpy as np
from sklearn.model_selection import KFold

from exp import Utility as Util, Result, Validation, AttackRunner, \
    ModelTraining, ModelScore, AttackScore


class Experiment:
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

    def fname(self):
        c = self.config
        v = "T" if c.validate else "F"
        return path.join(c.out, f'{c.name}_i{c.iter}_{v}.yaml')

    def get_conf(self, key: str):
        """Try get configration key, if exists."""
        return getattr(self.config, key) \
            if key and hasattr(self.config, key) else None

    def load_csv(self, ds_path: str, n_splits: int):
        """Read dataset and split into K-folds."""
        self.attrs, rows = Util.read_dataset(ds_path)
        self.X = rows[:, :-1]
        self.y = rows[:, -1].astype(int).flatten()
        self.folds = [x for x in KFold(
            n_splits=n_splits, shuffle=True).split(self.X)]
        for col_i in range(len(self.attrs) - 1):
            self.attr_ranges[col_i] = max(self.X[:, col_i])
        self.X = Util.normalize(self.X, self.attr_ranges)

    def run(self):
        """Run an experiment w/ classification, attack, validation."""
        conf = self.config
        self.start = time.time_ns()
        self.load_csv(conf.dataset, conf.folds)
        self.cls = ModelTraining(self.get_conf('xgb'))
        self.validation = Validation(conf.immutable, conf.constraints)
        self.attack = AttackRunner(
            *(conf.iter, conf.validate, self.get_conf('zoo')))
        Experiment.log_setup(self)
        for i, fold in enumerate(self.folds):
            self.exec_fold(i + 1, fold)
        self.end = time.time_ns()
        self.save_dependency_graph()
        Experiment.log_result(self.result)
        Util.write_result(self.fname(), self.to_dict())

    def exec_fold(self, fold_i: int, data_indices: List[int]):
        """Run one of K-folds."""
        Experiment.log_fold_num(fold_i)
        self.cls.reset(self.X.copy(), self.y.copy(),
                       *data_indices).train()
        self.result.append_cls(self.cls.score)
        Experiment.log_fold_model(self.cls.score)
        self.attack.reset(self.cls).run(self.validation)
        self.result.append_attack(self.attack.score)
        Experiment.log_fold_attack(self.attack.score)

    def save_dependency_graph(self):
        fn = path.join(self.config.out, self.config.name + '_graph.pdf')
        nn = dict([(i, n) for i, n in enumerate(self.attrs)])
        Util.plot_graph(self.validation.dep_graph, fn, node_names=nn)

    def to_dict(self) -> dict:
        return {'config': {
            'dataset': self.config.dataset,
            'config_path': self.config.config_path,
            'classifier': self.cls.name,
            'attack': self.attack.name,
            'n_classes': len(np.unique(self.y)),
            'n_attributes': len(self.attrs),
            'n_records': len(self.X),
            'attrs': self.attrs,
            'attr_ranges': dict([
                (str(k), float(v)) for k, v in
                zip(self.attrs, self.attr_ranges.values())]),
            'k_folds': self.config.folds,
            'max_iter': self.attack.max_iter,
            'immutable': self.validation.immutable,
            'constraints': list(self.validation.constraints.keys()),
            'dep_graph': dict([(k, list(v)) for k, v in
                               self.validation.desc.items()]),
            'predicates': self.get_conf('str_constraints'),
            'duration_sec': Util.time_sec(self.start, self.end)
        }, 'folds': {**self.result.to_dict()}}

    @staticmethod
    def log_setup(exp):
        Util.log('Dataset', exp.config.dataset)
        Util.log('Record count', len(exp.X))
        Util.log('Attributes', len(exp.attrs))
        Util.log('K-folds', exp.config.folds)
        Util.log('Classifier', exp.cls.name)
        Util.log('Classes', len(np.unique(exp.y)))
        Util.log('Immutable', len(exp.validation.immutable))
        Util.log('Constraints', len(exp.validation.constraints.keys()))
        Util.log('Attack', exp.attack.name)
        Util.log('Adv. validation', exp.config.validate)
        Util.log('Attack max iter', exp.attack.max_iter)

    @staticmethod
    def log_fold_num(fold_num: int):
        print('=' * 52)
        Util.log('Fold', fold_num)

    @staticmethod
    def log_fold_attack(asc: AttackScore):
        Util.logr('Evasions', asc.n_evasions, asc.n_records)
        Util.logr('Valid', asc.n_valid, asc.n_records)
        Util.logr('Valid+Evades', asc.n_valid_evades, asc.n_records)

    @staticmethod
    def log_fold_model(ms: ModelScore):
        Util.log('Accuracy', f'{ms.accuracy * 100:.2f} %')
        Util.log('Precision', f'{ms.precision * 100:.2f} %')
        Util.log('Recall', f'{ms.recall * 100:.2f} %')
        Util.log('F-score', f'{ms.f_score * 100:.2f} %')

    @staticmethod
    def log_result(res):
        print('=' * 52, '', '\nAVERAGE')
        Util.log('Accuracy', f'{res.accuracy.avg :.2f} %')
        Util.log('Precision', f'{res.precision.avg :.2f} %')
        Util.log('Recall', f'{res.recall.avg :.2f} %')
        Util.log('F-score', f'{res.f_score.avg :.2f} %')
        Util.logrd('Evasions', res.n_evasions.avg, res.n_records.avg)
        Util.logrd('Valid', res.n_valid.avg, res.n_records.avg)
        Util.logrd('Valid+Evades', res.n_ve.avg, res.n_records.avg)
