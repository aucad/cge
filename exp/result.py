from exp import Utility as Util


class Result(object):
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

    def append_attack(self, attack):
        self.n_evasions.append(attack.n_evasions)
        self.n_records.append(attack.n_records)

    def append_cls(self, cls):
        self.accuracy.append(cls.accuracy)
        self.precision.append(cls.precision)
        self.recall.append(cls.recall)
        self.f_score.append(cls.f_score)

    def to_dict(self):
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f_score': self.f_score,
            'n_records': self.n_records,
            'n_evasions': self.n_evasions}
