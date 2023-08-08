import numpy as np

from exp import Loggable, CONSTR_DICT, categorize
from exp.utility import sdiv, log, logr, logrd, attr_of


def score_valid(ori: np.ndarray, adv: np.ndarray, cd: CONSTR_DICT, scalars):
    """Record validity scoring"""
    immutable, single_ft, multi_ft = categorize(cd)
    invalid = np.array([], dtype=int)
    for ft_i in immutable:
        correct, modified = ori[:, ft_i], adv[:, ft_i]
        wrong = np.where(np.subtract(correct, modified) != 0)[0]
        invalid = np.union1d(invalid, wrong)
    for ft_i in single_ft:
        pred, in_, scale = cd[ft_i][1], adv[:, ft_i], scalars[ft_i]
        bits = np.vectorize(pred)(in_ * scale)
        wrong = np.where(bits == 0)[0]
        invalid = np.union1d(invalid, wrong)
    for (sources, pred) in [cd[ft_i] for ft_i in multi_ft]:
        in_, sf = adv[:, sources], scalars[list(sources)]
        inputs = np.multiply(in_, sf)
        bits = np.apply_along_axis(pred, 1, inputs)
        wrong = np.where(bits == 0)[0]
        invalid = np.union1d(invalid, wrong)
    return ori.shape[0] - invalid.shape[0], invalid


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
        self.accuracy = sdiv(tp_tn, len(lbl), -1, False)
        self.precision = p = sdiv(tp, tp_fp, 1, False)
        self.recall = r = sdiv(tp, tp_fn, 1, False)
        self.f_score = sdiv(2 * p * r, p + r, 0, False)

    def log(self):
        for a in attr_of(self, (int, float)):
            log(a.capitalize(), f'{getattr(self, a) * 100:.2f} %')


class AttackScore(Loggable):

    def __init__(self):
        self.evasions = None
        self.valid_evades = None
        self.n_evasions = 0
        self.n_records = 0
        self.n_valid = 0
        self.n_valid_evades = 0

    def calculate(self, attack, constraints, amax):
        ori_x, ori_y = attack.ori_x, attack.ori_y
        adv_x, adv_y = attack.adv_x, attack.adv_y
        original = attack.cls.predict(ori_x, ori_y)
        correct = np.where(ori_y == original)[0]
        evades = np.where(adv_y != original)[0]
        self.evasions = np.intersect1d(evades, correct)
        self.n_valid, inv_idx = score_valid(ori_x, adv_x, constraints, amax)
        self.valid_evades = np.setdiff1d(self.evasions, inv_idx)
        self.n_evasions = len(self.evasions)
        self.n_valid_evades = len(self.valid_evades)
        self.n_records = ori_x.shape[0]

    def log(self):
        logr('Evasions', self.n_evasions, self.n_records)
        logr('Valid', self.n_valid, self.n_records)
        logr('Valid+Evades', self.n_valid_evades, self.n_records)


class Result(Loggable):
    class AvgList(list):

        @property
        def avg(self):
            return sdiv(sum(self), len(self))

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
        for a in attr_of(obj, (int, float)):
            if a in dir(self):
                getattr(self, a).append(getattr(obj, a))

    def to_dict(self):
        return dict([(str(a), getattr(self, a).to_dict())
                     for a in attr_of(self, Result.AvgList)])

    def log(self):
        print('=' * 52, '\nAVERAGE')
        log('Accuracy', f'{self.accuracy.avg :.2f} %')
        log('Precision', f'{self.precision.avg :.2f} %')
        log('Recall', f'{self.recall.avg :.2f} %')
        log('F-score', f'{self.f_score.avg :.2f} %')
        logrd('Evasions', self.n_evasions.avg, self.n_records.avg)
        logrd('Valid', self.n_valid.avg, self.n_records.avg)
        logrd('Valid Evasions',
              self.n_valid_evades.avg, self.n_records.avg)
