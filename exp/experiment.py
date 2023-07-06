import sys
import os
from collections import namedtuple

import numpy as np
from art.estimators.classification import XGBoostClassifier
from sklearn.model_selection import KFold
from xgboost import DMatrix, train as xg_train

from exp import MyZooAttack as ZooAttack, Utility as Util, Result


# TODO: apply experiment with validation
# TODO: setup experiment to measure cases w & w/o validation

class Experiment:

    def __init__(self, conf):
        c_keys = ",".join(list(conf.keys()))
        self.X = None
        self.y = None
        self.cls = None
        self.folds = None
        self.attack = None
        self.attrs = []
        self.mask_cols = []
        self.attr_ranges = {}
        self.config = (namedtuple('exp', c_keys)(**conf))
        self.stats = Result()

    @property
    def n_records(self) -> int:
        return len(self.X)

    def custom_config(self, key):
        return getattr(self.config, key) \
            if key and hasattr(self.config, key) else None

    def load_csv(self, ds_path: str, n_splits: int):
        self.attrs, rows = Util.read_dataset(ds_path)
        self.X = rows[:, :-1]
        self.y = rows[:, -1].astype(int).flatten()
        self.mask_cols, n_feat = [], len(self.attrs) - 1
        self.folds = [x for x in KFold(
            n_splits=n_splits, shuffle=True).split(self.X)]
        for col_i in range(n_feat):
            self.attr_ranges[col_i] = max(self.X[:, col_i])
            col_values = list(np.unique(self.X[:, col_i]))
            if set(col_values).issubset({0, 1}):
                self.mask_cols.append(col_i)

    def run(self):
        config = self.config
        self.load_csv(config.dataset, config.folds)
        self.cls = XGBoostRunner(*(
            config.out, self.attrs, self.y,
            self.mask_cols, self.attr_ranges,
            self.custom_config('xgb')))
        self.attack = ZooRunner(*(
            config.iter, self.custom_config('zoo')))
        self.log_experiment_setup()
        for i, fold in enumerate(self.folds):
            self.exec_fold(i + 1, fold)
        self.log_experiment_result()
        Util.write_result(self.config.out, self.to_dict())

    def exec_fold(self, fi, f_idx):
        self.cls.reset() \
            .load(self.X.copy(), self.y.copy(), *f_idx, fi) \
            .train()
        self.stats.append_cls(self.cls)
        self.log_training_result(fi)
        self.attack.reset().set_cls(self.cls).run().eval()
        self.stats.append_attack(self.attack)
        self.log_fold_attack()

    def log_experiment_setup(self):
        Util.log('Dataset', self.config.dataset)
        Util.log('Record count', len(self.X))
        Util.log('Attributes', len(self.attrs))
        Util.log('K-folds', self.config.folds)
        Util.log('Classifier', self.cls.name)
        Util.log('Classes', ", ".join(self.cls.class_names))
        Util.log('Mutable', len(self.cls.mutable))
        Util.log('Immutable', len(self.cls.immutable))
        Util.log('Attack', self.attack.name)
        Util.log('Attack max iter', self.attack.max_iter)

    def log_training_result(self, fold_n: int):
        print('=' * 52)
        Util.log('Fold', fold_n)
        Util.log('Accuracy', f'{self.cls.accuracy * 100:.2f} %')
        Util.log('Precision', f'{self.cls.precision * 100:.2f} %')
        Util.log('Recall', f'{self.cls.recall * 100:.2f} %')
        Util.log('F-score', f'{self.cls.f_score * 100:.2f} %')

    def log_fold_attack(self):
        Util.logr('Evasions', self.attack.n_evasions, self.attack.n_records)

    def log_experiment_result(self):
        print('=' * 52, '', '\nAVERAGE')
        Util.log('Accuracy', f'{self.stats.accuracy.avg :.2f} %')
        Util.log('Precision', f'{self.stats.precision.avg :.2f} %')
        Util.log('Recall', f'{self.stats.recall.avg :.2f} %')
        Util.log('F-score', f'{self.stats.f_score.avg :.2f} %')
        Util.logr('Evasions', self.stats.n_evasions.avg / 100,
                  self.stats.n_records.avg / 100)

    def to_dict(self) -> dict:
        return {
            'dataset': self.config.dataset,
            'n_records': len(self.X),
            'n_attributes': len(self.attrs),
            'attrs': self.attrs,
            'immutable': self.mask_cols,
            'attr_mutable': self.cls.mutable,
            'attr_immutable': self.cls.immutable,
            'classes': self.cls.class_names,
            'k_folds': self.config.folds,
            'classifier': self.config.cls,
            'attack': self.config.attack,
            'attr_ranges': dict(
                zip(self.attrs, self.attr_ranges.values())),
            'max_iter': self.attack.max_iter,
            **self.stats.to_dict()
        }


class XGBoostRunner:

    def __init__(self, out, attrs, y, masks, ranges, conf):
        self.name = 'xgboost'
        self.out_dir = out
        self.attrs = attrs[:]
        self.n_features = len(self.attrs) - 1
        self.classes = np.unique(y)
        self.attr_ranges = ranges
        self.mask_cols = masks
        self.classifier = None
        self.model = None
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.fold_n = 1
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f_score = 0
        self.reset()
        self.cls_conf = conf or {}

    def reset(self):
        self.classifier = None
        self.model = None
        self.train_x = np.array([])
        self.train_y = np.array([])
        self.test_x = np.array([])
        self.test_y = np.array([])
        self.fold_n = 1
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f_score = 0
        return self

    @property
    def class_names(self):
        return [('malicious' if cn == 1 else 'benign')
                for cn in self.classes]

    @property
    def mutable(self):
        return sorted([
            a for i, a in enumerate(self.attrs)
            if i not in self.mask_cols and i < self.n_features])

    @property
    def immutable(self):
        return sorted([
            a for i, a in enumerate(self.attrs)
            if i in self.mask_cols and i < self.n_features])

    def load(self, x, y, fold_train, fold_test, fold_n):
        self.train_x = self.normalize(x[fold_train, :])
        self.train_y = y[fold_train].astype(int).flatten()
        self.test_x = self.normalize(x[fold_test, :])
        self.test_y = y[fold_test].astype(int).flatten()
        self.classes = np.unique(y)
        self.fold_n = fold_n
        return self

    @staticmethod
    def formatter(x, y):
        return DMatrix(x, y)

    def predict(self, data):
        tmp = self.model.predict(data)
        ax = 1 if len(tmp.shape) == 2 else 0
        return tmp.argmax(axis=ax)

    def init_learner(self) -> None:
        d_train = self.formatter(self.train_x, self.train_y)
        sys.stdout = open(os.devnull, 'w')
        self.model = xg_train(
            num_boost_round=20,
            dtrain=d_train,
            evals=[(d_train, 'eval'), (d_train, 'train')],
            params={'num_class': len(self.classes), **self.cls_conf})
        sys.stdout = sys.__stdout__  # re-enable print
        self.classifier = XGBoostClassifier(
            model=self.model,
            nb_features=self.n_features,
            nb_classes=len(self.classes),
            clip_values=(0, 1))

    def train(self):
        self.init_learner()
        records = (
            (self.test_x, self.test_y) if len(self.test_x) > 0
            else (self.train_x, self.train_y))
        predictions = self.predict(self.formatter(*records))
        self.score(records[1], predictions)
        return self

    def normalize(self, data):
        np.seterr(divide='ignore', invalid='ignore')
        for i in range(self.n_features):
            range_max = self.attr_ranges[i]
            data[:, i] = (data[:, i]) / range_max
            data[:, i] = np.nan_to_num(data[:, i])
        return data

    def score(self, true_labels, predictions, positive=0):
        """Calculate performance metrics."""
        tp, tp_tn, p_pred, p_actual = 0, 0, 0, 0
        for actual, pred in zip(true_labels, predictions):
            int_pred = int(round(pred, 0))
            if int_pred == positive:
                p_pred += 1
            if actual == positive:
                p_actual += 1
            if int_pred == actual:
                tp_tn += 1
            if int_pred == actual and int_pred == positive:
                tp += 1
        self.accuracy = Util.sdiv(tp_tn, len(predictions), -1, False)
        self.precision = Util.sdiv(tp, p_pred, 1, False)
        self.recall = Util.sdiv(tp, p_actual, 1, False)
        self.f_score = Util.sdiv(
            2 * self.precision * self.recall,
            self.precision + self.recall, 0, False)


class ZooRunner:

    def __init__(self, i, conf):
        self.name = 'zoo'
        self.max_iter = 10 if i < 1 else i
        self.cls = None
        self.ori_x = None
        self.ori_y = None
        self.adv_x = None
        self.adv_y = None
        self.evasions = None
        self.reset()
        self.attack_conf = conf or {}

    def reset(self):
        self.cls = None
        self.ori_x = np.array([])
        self.ori_y = np.array([])
        self.adv_x = np.array([])
        self.adv_y = np.array([])
        self.evasions = np.array([])
        return self

    @property
    def n_records(self):
        return len(self.ori_x)

    @property
    def n_evasions(self):
        return len(self.evasions)

    def set_cls(self, cls):
        indices = range(len(cls.test_x))
        self.cls = cls
        self.ori_x = cls.test_x.copy()[indices, :]
        self.ori_y = cls.test_y.copy()[indices]
        return self

    def eval(self):
        ori_in = self.cls.formatter(self.ori_x, self.ori_y)
        original = self.cls.predict(ori_in).flatten().tolist()
        correct = np.array((np.where(
            np.array(self.ori_y) == original)[0]).flatten().tolist())
        adv_in = self.cls.formatter(self.adv_x, self.ori_y)
        adversarial = self.cls.predict(adv_in).flatten().tolist()
        self.adv_y = np.array(adversarial)
        evades = np.array((np.where(
            self.adv_y != original)[0]).flatten().tolist())
        self.evasions = np.intersect1d(evades, correct)

    def run(self):
        self.adv_x = ZooAttack(**{
            'classifier': self.cls.classifier,
            **self.attack_conf}) \
            .generate(x=self.ori_x)
        return self
