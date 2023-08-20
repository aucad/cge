import sys

import numpy as np
from art.attacks.evasion import ZooAttack, AutoProjectedGradientDescent

from exp import ZooConst, PGDConst, AttackScore, Validation, Validatable


class AttackRunner:
    """Wrapper for adversarial attack"""

    def __init__(self, attack_type, apply_constr, conf):
        if attack_type == 'zoo':
            self.attack = ZooConst if apply_constr else ZooAttack
        elif attack_type == 'pgd':
            self.attack = PGDConst if apply_constr else \
                AutoProjectedGradientDescent
        self.name = self.attack.__name__
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
        sys.stdout.write('\x1b[1A')
        sys.stdout.write('\x1b[2K')
        self.adv_y = np.array(self.cls.predict(
            self.adv_x, self.ori_y).flatten())
        self.score.calculate(self, v_model.constraints, v_model.scalars)
        return self

    def to_dict(self):
        return {'name': self.name, 'config': self.conf,
                'can_validate': self.can_validate}
