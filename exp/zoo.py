import numpy as np
from art.attacks.evasion import ZooAttack

from exp.types import Validatable


class ZooConst(ZooAttack, Validatable):

    def _generate_batch(
            self, x_batch: np.ndarray, y_batch: np.ndarray
    ) -> np.ndarray:
        o_best_attack = super()._generate_batch(x_batch, y_batch)
        return self.v_model.enforce(x_batch, o_best_attack)
