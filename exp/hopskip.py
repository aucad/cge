from typing import Optional

import numpy as np
from art.attacks.evasion import HopSkipJump

from exp import Validatable


class HopSkipJumpConst(HopSkipJump, Validatable):

    def _attack(
            self,
            initial_sample: np.ndarray,
            original_sample: np.ndarray,
            target: int,
            mask: Optional[np.ndarray],
            clip_min: float,
            clip_max: float,
    ) -> np.ndarray:
        x_adv = super()._attack(
            initial_sample, original_sample,
            target, mask, clip_min, clip_max)

        # adjust shape: 1d -> 2d -> 1d
        return self.v_model.enforce(
            np.array([original_sample]),
            np.array([x_adv]))[0]  # NEW
