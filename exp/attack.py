import sys

import numpy as np
from art.attacks.evasion import ZooAttack, \
    ProjectedGradientDescent, HopSkipJump

from comparison import cpgd_apply_predict, CPGD
from exp import ZooConst, PGDConst, HopSkipJumpConst, AttackScore, \
    Validation, Validatable


class AttackPicker:
    ZOO = 'zoo'
    HSJ = 'hsj'
    PDG = 'pgd'
    CPGD = 'cpgd'

    @staticmethod
    def list_attacks():
        return sorted([
            AttackPicker.ZOO, AttackPicker.PDG,
            AttackPicker.HSJ, AttackPicker.CPGD])

    @staticmethod
    def load_attack(attack_name, apply_constr: bool):

        if attack_name == AttackPicker.ZOO:
            return ZooConst if apply_constr else ZooAttack

        if attack_name == AttackPicker.PDG:
            return PGDConst if apply_constr else \
                ProjectedGradientDescent

        if attack_name == AttackPicker.HSJ:
            return HopSkipJumpConst if apply_constr else HopSkipJump

        if attack_name == AttackPicker.CPGD:
            return CPGD


class AttackRunner:
    """Wrapper for adversarial attack"""

    def __init__(self, kind: str, constr: bool, conf):
        self.attack = AttackPicker.load_attack(kind, constr)
        self.name = self.attack.__name__
        self.cls = None
        self.ori_x = None
        self.ori_y = None
        self.adv_x = None
        self.adv_y = None
        self.score = None
        self.conf = conf or {}
        self.apply_constr = constr

    def reset(self, cls):
        self.cls = cls
        self.ori_y = cls.test_y.copy()
        self.ori_x = cls.test_x.copy()
        self.adv_x = None
        self.adv_y = None
        self.score = AttackScore()
        return self

    @property
    def can_validate(self):
        return issubclass(self.attack, Validatable)

    def run(self, v_model: Validation):
        """Generate adversarial examples and score."""
        if issubclass(self.attack, CPGD):
            self.adv_x, self.adv_y = cpgd_apply_predict(
                self.cls.model, self.ori_x, self.ori_y,
                self.apply_constr, **self.conf)
        else:
            aml_attack = self.attack(self.cls.classifier, **self.conf)
            if self.can_validate:
                aml_attack.vhost().v_model = v_model
            self.adv_x = aml_attack.generate(x=self.ori_x)
            self.adv_y = np.array(self.cls.predict(
                self.adv_x, self.ori_y).flatten())
        sys.stdout.write('\x1b[1A')
        sys.stdout.write('\x1b[2K')
        self.score.calculate(
            self, v_model.constraints, v_model.scalars)
        return self

    def to_dict(self):
        return {'name': self.name, 'config': self.conf,
                'can_validate': self.can_validate}
