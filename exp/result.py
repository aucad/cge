from abc import ABC, abstractmethod

import numpy as np

from exp import Utility as Util


class Loggable(ABC):
    @abstractmethod
    def log(self):
        pass

    @staticmethod
    def attr_of(o, t):
        return [x for x in dir(o) if isinstance(getattr(o, x), t)]


class ModelScore(Loggable):

    def __init__(self):
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f_score = 0

    def calculate(self, true_labels, predictions, positive=0):
        """Calculate classifier performance metrics."""
        prd = np.array([round(p, 0) for p in predictions], dtype=int)
        lbl = np.array(true_labels)
        tp_tn = int(np.sum(prd == lbl))
        tp_fp = int(np.sum(prd == positive))
        tp_fn = int(np.sum(lbl == positive))
        tp = len(np.where((prd == lbl) & (lbl == positive))[0])
        self.accuracy = Util.sdiv(tp_tn, len(lbl), -1, False)
        self.precision = p = Util.sdiv(tp, tp_fp, 1, False)
        self.recall = r = Util.sdiv(tp, tp_fn, 1, False)
        self.f_score = Util.sdiv(2 * p * r, p + r, 0, False)

    def log(self):
        for a in self.attr_of(self, (int, float)):
            Util.log(a.capitalize(), f'{getattr(self, a) * 100:.2f} %')


class AttackScore(Loggable):

    def __init__(self):
        self.evasions = None
        self.valid_evades = None
        self.n_evasions = 0
        self.n_records = 0
        self.n_valid = 0
        self.n_valid_evades = 0

    def calculate(self, attack, validation):
        ori_x, ori_y = attack.ori_x, attack.ori_y
        adv_x, adv_y = attack.adv_x, attack.adv_y
        original = attack.cls.predict(ori_x, ori_y)
        correct = np.where(ori_y == original)[0]
        evades = np.where(adv_y != original)[0]
        self.n_valid, v_idx = validation.score_valid(ori_x, adv_x)
        self.evasions = np.intersect1d(evades, correct)
        self.valid_evades = np.intersect1d(
            self.evasions, np.where(v_idx == 0)[0])
        self.n_evasions = len(self.evasions)
        self.n_valid_evades = len(self.valid_evades)
        self.n_records = ori_x.shape[0]

    def log(self):
        Util.logr('Evasions', self.n_evasions, self.n_records)
        Util.logr('Valid', self.n_valid, self.n_records)
        Util.logr('Valid+Evades', self.n_valid_evades, self.n_records)


class Result(Loggable):
    class AvgList(list):

        @property
        def avg(self):
            return Util.sdiv(sum(self), len(self))

        def to_dict(self):
            return [round(x, 6) for x in list(self)]

    def __init__(self):
        self.accuracy = Result.AvgList()
        self.precision = Result.AvgList()
        self.recall = Result.AvgList()
        self.f_score = Result.AvgList()
        self.n_records = Result.AvgList()
        self.n_evasions = Result.AvgList()
        self.n_valid = Result.AvgList()
        self.n_valid_evades = Result.AvgList()

    def append(self, obj):
        for a in Result.attr_of(obj, (int, float)):
            if a in dir(self):
                getattr(self, a).append(getattr(obj, a))

    def to_dict(self):
        return dict([(str(a), getattr(self, a).to_dict())
                     for a in Result.attr_of(self, Result.AvgList)])

    def log(self):
        print('=' * 52, '\nAVERAGE')
        Util.log('Accuracy', f'{self.accuracy.avg :.2f} %')
        Util.log('Precision', f'{self.precision.avg :.2f} %')
        Util.log('Recall', f'{self.recall.avg :.2f} %')
        Util.log('F-score', f'{self.f_score.avg :.2f} %')
        Util.logrd('Evasions', self.n_evasions.avg, self.n_records.avg)
        Util.logrd('Valid', self.n_valid.avg, self.n_records.avg)
        Util.logrd('Valid Evasions',
                   self.n_valid_evades.avg, self.n_records.avg)
