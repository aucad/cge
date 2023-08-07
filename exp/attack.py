import numpy as np

from art.attacks.evasion import ZooAttack

from exp import ZooConst, AttackScore, Validation
from exp.types import Validatable


class AttackRunner:
    """Wrapper for adversarial attack"""

    def __init__(self, i, apply_constr, conf):
        self.attack = ZooConst if apply_constr else ZooAttack
        self.name = self.attack.__name__
        self.max_iter = 10 if i < 1 else i
        self.cls = None
        self.ori_x = None
        self.ori_y = None
        self.adv_x = None
        self.adv_y = None
        self.score = None
        self.conf = conf or {}

    def reset(self, cls):
        self.cls = cls
        self.ori_x = cls.test_x.copy()
        self.ori_y = cls.test_y.copy()
        self.adv_x = None
        self.adv_y = None
        self.score = AttackScore()
        return self

    @property
    def can_validate(self):
        return issubclass(self.attack, Validatable)

    def run(self, v_model: Validation):
        """Generate adversarial examples and score."""
        aml_attack = self.attack(self.cls.classifier, **self.conf)
        if self.can_validate:
            aml_attack.v_model = v_model
        self.adv_x = aml_attack.generate(x=self.ori_x)
        self.adv_y = np.array(self.cls.predict(
            self.adv_x, self.ori_y).flatten())
        self.score.calculate(self, v_model)
        return self
