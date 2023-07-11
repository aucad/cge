import numpy as np
from exp import Utility as Util


class ModelScore:
    """Classifier scoring."""

    def __init__(self):
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f_score = 0

    def calculate(self, true_labels, predictions, positive=0):
        """Calculate classifier performance metrics."""
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


class AttackScore:

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
        ori_in = attack.cls.formatter(ori_x, ori_y)
        original = attack.cls.predict(ori_in).flatten().tolist()
        correct = np.array((np.where(
            np.array(ori_y) == original)[0]).flatten().tolist())
        evades = np.array((np.where(
            adv_y != original)[0]).flatten().tolist())
        self.evasions = np.intersect1d(evades, correct)
        self.n_evasions = len(self.evasions)
        self.n_valid, idx_valid = validation.score_valid(adv_x)
        valid = np.array((np.where(
            idx_valid == 0)[0]).flatten().tolist())
        self.valid_evades = np.intersect1d(self.evasions, valid)
        self.n_valid_evades = len(self.valid_evades)
        self.n_records = ori_x.shape[0]


class Result(object):
    """Experiment result."""

    class AvgList(list):

        @property
        def avg(self):
            return Util.sdiv(sum(self), len(self))

    def __init__(self):
        self.accuracy = Result.AvgList()
        self.precision = Result.AvgList()
        self.recall = Result.AvgList()
        self.f_score = Result.AvgList()
        self.n_records = Result.AvgList()
        self.n_evasions = Result.AvgList()
        self.n_valid = Result.AvgList()
        self.n_ve = Result.AvgList()

    def append_attack(self, asc: AttackScore):
        self.n_evasions.append(asc.n_evasions)
        self.n_records.append(asc.n_records)
        self.n_valid.append(asc.n_valid)
        self.n_ve.append(asc.n_valid_evades)

    def append_cls(self, ms: ModelScore):
        self.accuracy.append(ms.accuracy)
        self.precision.append(ms.precision)
        self.recall.append(ms.recall)
        self.f_score.append(ms.f_score)

    def to_dict(self):
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f_score': self.f_score,
            'n_records': self.n_records,
            'n_valid_evades': self.n_ve,
            'n_evasions': self.n_evasions,
            'n_valid': self.n_valid,
        }
