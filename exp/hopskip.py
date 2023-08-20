from typing import Optional

import numpy as np
from art.attacks.evasion import HopSkipJump

from exp import Validatable


class HopSkipConst(HopSkipJump, Validatable):

    def generate(
            self, x: np.ndarray, y: Optional[np.ndarray] = None,
            **kwargs
    ) -> np.ndarray:
        o_best_attack = super().generate(x, y)
        return self.v_model.enforce(x, o_best_attack)
