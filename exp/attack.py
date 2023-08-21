import sys

import numpy as np
from art.attacks.evasion import ZooAttack, \
    ProjectedGradientDescent, HopSkipJump

from exp import ZooConst, PGDConst, HopSkipConst, \
    AttackScore, Validation, Validatable


class AttackPicker:
    ZOO = 'zoo'
    PDG = 'pgd'
    HSJ = 'hsj'

    @staticmethod
    def list_attacks():
        return sorted([AttackPicker.ZOO, AttackPicker.PDG,
                       AttackPicker.HSJ])

    @staticmethod
    def load_attack(attack_name, apply_constr: bool):

        if attack_name == AttackPicker.ZOO:
            return ZooConst if apply_constr else ZooAttack

        if attack_name == AttackPicker.PDG:
            return PGDConst if apply_constr else \
                ProjectedGradientDescent

        if attack_name == AttackPicker.HSJ:
            return HopSkipConst if apply_constr else HopSkipJump


class AttackRunner:
    """Wrapper for adversarial attack"""

    def __init__(self, attack_type: str, apply_constr: bool, conf):
        self.attack = AttackPicker.load_attack(
            attack_type, apply_constr)
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
        self.score.calculate(
            self, v_model.constraints, v_model.scalars)
        return self

    def to_dict(self):
        return {'name': self.name, 'config': self.conf,
                'can_validate': self.can_validate}
