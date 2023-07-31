import numpy as np

from art.attacks.evasion import ZooAttack as ZooUniversal

from exp import ZooConst, AttackScore, Validation, Validatable


class AttackRunner:
    """Wrapper for zoo attack"""

    def __init__(self, i, v, conf):
        self.mode = ZooConst if v else ZooUniversal
        self.name = 'Zoo+Validation' if v else 'Zoo Basic'
        self.max_iter = 10 if i < 1 else i
        self.cls = None
        self.ori_x = None
        self.ori_y = None
        self.adv_x = None
        self.adv_y = None
        self.score = None
        self.attack_conf = conf or {}
        self.n_records = 0

    def reset(self, cls):
        indices = range(len(cls.test_x))
        self.cls = cls
        self.ori_x = cls.test_x.copy()[indices, :]
        self.ori_y = cls.test_y.copy()[indices]
        self.n_records = len(self.ori_x)
        self.adv_x = np.array([])
        self.adv_y = np.array([])
        self.score = AttackScore()
        return self

    def run(self, v_model: Validation):
        """Generate adversarial examples and score."""
        attack = self.mode(
            **{'classifier': self.cls.classifier,
               **self.attack_conf})
        if issubclass(self.mode, Validatable):
            attack.set_validation(v_model)
        self.adv_x = attack.generate(x=self.ori_x)
        self.adv_y = np.array(self.cls.predict(self.cls.formatter(
            self.adv_x, self.ori_y)).flatten().tolist())
        self.score.calculate(self, v_model)
        return self
