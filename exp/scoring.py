import numpy as np
import sklearn.metrics as skm

from exp import CONSTR_DICT as CD, Validation
from exp.utility import sdiv, log, logr, logd, attr_of, dur_sec


def score_valid(ori: np.ndarray, adv: np.ndarray, cd: CD, scalars):
    """Adversarial example validity scoring"""
    immutable, mutable = Validation.categorize(cd)
    invalid = np.array([], dtype=int)
    for ft_i in immutable:
        correct, modified = ori[:, ft_i], adv[:, ft_i]
        invalid = np.where(np.subtract(correct, modified) != 0)[0]
    for (sources, pred) in [cd[ft_i] for ft_i in mutable]:
        in_ = adv[:, sources]
        for i, ft_i in enumerate(sources):
            (mn, mx) = scalars[ft_i]
            in_[:, i] = (in_[:, i] * (mx - mn)) + mn
        bits = np.apply_along_axis(pred, 1, in_)
        invalid = np.union1d(invalid, np.where(bits == 0)[0])
    return ori.shape[0] - invalid.shape[0], invalid


class ModelScore:

    def __init__(self):
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f_score = 0

    def calculate(self, true_labels, predictions):
        """Calculate classifier performance metrics."""
        true_labels = np.array(true_labels)
        prd = np.array([round(p, 0) for p in predictions], dtype=int)
        self.accuracy = float(skm.accuracy_score(true_labels, prd))
        self.precision = float(skm.precision_score(true_labels, prd))
        self.recall = float(skm.recall_score(true_labels, prd))
        self.f_score = float(skm.f1_score(true_labels, prd))

    def log(self):
        for a in attr_of(self, (int, float)):
            log(a.capitalize().replace('_', '-'),
                f'{getattr(self, a) * 100:.2f} %')


class AttackScore:

    def __init__(self):
        self.evasions = None
        self.valid_evades = None
        self.n_evasions = 0
        self.n_records = 0
        self.n_valid = 0
        self.n_valid_evades = 0
        self.dur = 0

    def calculate(self, attack, constraints, attr_range, dur):
        ori_x, ori_y = attack.ori_x, attack.ori_y
        adv_x, adv_y = attack.adv_x, attack.adv_y
        original = attack.cls.predict(ori_x, ori_y)
        correct = np.where(ori_y == original)[0]
        evades = np.where(adv_y != original)[0]
        self.evasions = np.intersect1d(evades, correct)
        self.n_valid, inv_idx = \
            score_valid(ori_x, adv_x, constraints, attr_range)
        self.valid_evades = np.setdiff1d(self.evasions, inv_idx)
        self.n_evasions = len(self.evasions)
        self.n_valid_evades = len(self.valid_evades)
        self.n_records = ori_x.shape[0]
        self.dur = dur

    def log(self):
        logr('Evasions', self.n_evasions, self.n_records)
        logr('Valid', self.n_valid, self.n_records)
        logr('Valid+Evades', self.n_valid_evades, self.n_records)
        log('Attack Duration', f'{dur_sec(self.dur) :.2f} s')


class Result:
    class AvgList(list):

        @property
        def avg(self):
            return sdiv(sum(self), len(self))

        @property
        def abs_avg(self):
            return sdiv(sum(self), len(self), mult=False)

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
        self.dur = Result.AvgList()

    def append(self, obj):
        for a in attr_of(obj, (int, float)):
            if a in dir(self):
                getattr(self, a).append(getattr(obj, a))

    def to_dict(self):
        return dict([(str(a), getattr(self, a).to_dict())
                     for a in attr_of(self, Result.AvgList)])

    def log(self):
        print('=' * 50, '\nAVERAGE')
        log('Accuracy', f'{self.accuracy.avg :.2f} %')
        log('Precision', f'{self.precision.avg :.2f} %')
        log('Recall', f'{self.recall.avg :.2f} %')
        log('F-score', f'{self.f_score.avg :.2f} %')
        logd('Evasions', self.n_evasions.avg, self.n_records.avg)
        logd('Valid', self.n_valid.avg, self.n_records.avg)
        logd('Valid Evasions',
             self.n_valid_evades.avg, self.n_records.avg)
        log('Attack Duration', f'{dur_sec(self.dur.abs_avg) :.2f} s')
