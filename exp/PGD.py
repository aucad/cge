import numpy as np
from art.attacks.evasion import AutoProjectedGradientDescent

from exp import Validatable


class PGDConst(AutoProjectedGradientDescent, Validatable):

    def generate(
            self, x_batch: np.ndarray, y_batch: np.ndarray = None,
            **kwargs
    ) -> np.ndarray:
        x_adv = super().generate(x_batch, y_batch)
        return self.v_model.enforce(x_batch, x_adv)
