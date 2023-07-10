import numpy as np

from exp import MyZooAttack as ZooAttack


class AttackRunner:

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
