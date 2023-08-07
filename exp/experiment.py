import time
from collections import namedtuple

import numpy as np
from sklearn.model_selection import KFold

from exp import \
    Result, Validation, AttackRunner, ModelTraining, Loggable, score_valid
from exp.preproc import read_dataset, normalize
from exp.utility import log, plot_graph, time_sec, write_result, fname


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
        self.o_valid = None
        self.ov_idx = None

    def conf(self, key: str):
        """Try get configration key."""
        if key and hasattr(self.config, key):
            return getattr(self.config, key)

    def run(self):
        """Run an experiment of K folds."""
        c = self.config
        self.attrs, rows = read_dataset(c.dataset)
        self.X = rows[:, :-1]
        self.y = rows[:, -1].astype(int).flatten()
        self.X = normalize(self.X, [
            max(self.X[:, i]) for i in range(len(self.attrs) - 1)])
        self.validate_original()
        self.folds = [x for x in KFold(
            n_splits=c.folds, shuffle=True).split(self.X)]
        self.cls = ModelTraining(self.conf('xgb'))
        self.attack = AttackRunner(c.iter, c.validate, self.conf('zoo'))
        self.validation = Validation(c.constraints)

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
            log('Fold #', fold_num + 1)
            self.cls.score.log()
            self.attack.score.log()
        self.end = time.time_ns()
        self.result.log()

        plot_graph(self.validation, c, self.attrs)
        write_result(fname(c), self.to_dict())

    def validate_original(self):
        """ensure only initially valid examples are included in input."""
        o_valid, self.ov_idx = score_valid(
            self.X.copy(), self.X.copy(), self.config.constraints)
        self.o_valid = self.X.shape[0] - o_valid
        if self.o_valid > 0:
            self.X = np.delete(self.X, self.ov_idx, 0)
            self.y = np.delete(self.y, self.ov_idx, 0)
            print(f'WARNING: {self.o_valid} invalid entries were excluded')

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
            'description': self.config.desc,
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
        }, 'validation': {
            **({'original_dataset_invalid_rows': self.ov_idx.tolist()}
               if self.o_valid > 0 else {}),
            'constraints': list(self.validation.constraints.keys()),
            'constraints_enforced': self.attack.can_validate,
            'predicates_immutable': self.validation.immutable,
            'predicates': dict(
                [(str(k), str(v).strip()) for (k, v) in
                 self.conf('str_constraints').items()]),
            'predicates_sing+multi': dict(
                [(k, str(v) if isinstance(v, str) else [
                    str(x).strip() if isinstance(x, str) else x
                    for x in v]) for k, v
                 in self.conf('str_func').items()]),
            'dependencies': dict(
                [(k, list(v)) for k, v in
                 self.validation.desc.items() if len(v) > 0]),
        }, 'folds': {**self.result.to_dict()}}
