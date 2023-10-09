from __future__ import absolute_import, division, print_function, \
    unicode_literals

import logging
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
# noinspection PyPackageRequirements
from art.attacks.evasion import ProjectedGradientDescent, \
    ProjectedGradientDescentNumpy
# noinspection PyPackageRequirements
from art.summary_writer import SummaryWriter

from exp import Validatable

if TYPE_CHECKING:
    # noinspection PyPackageRequirements
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE, \
        OBJECT_DETECTOR_TYPE

logger = logging.getLogger(__name__)


class PGDNumpyConstr(ProjectedGradientDescentNumpy, Validatable):

    def _compute(
            self,
            x: np.ndarray,
            x_init: np.ndarray,
            y: np.ndarray,
            mask: Optional[np.ndarray],
            eps: Union[int, float, np.ndarray],
            eps_step: Union[int, float, np.ndarray],
            project: bool,
            random_init: bool,
            batch_id_ext: Optional[int] = None,
            decay: Optional[float] = None,
            momentum: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        x_adv = super()._compute(
            x, x_init, y, mask, eps, eps_step, project,
            random_init, batch_id_ext, decay, momentum)

        return self.v_model.enforce(x, x_adv)  # NEW


class VPGD(ProjectedGradientDescent, Validatable):

    def __init__(
            self,
            estimator: Union["CLASSIFIER_LOSS_GRADIENTS_TYPE",
                             "OBJECT_DETECTOR_TYPE"],
            norm: Union[int, float, str] = np.inf,
            eps: Union[int, float, np.ndarray] = 0.3,
            eps_step: Union[int, float, np.ndarray] = 0.1,
            decay: Optional[float] = None,
            max_iter: int = 100,
            targeted: bool = False,
            num_random_init: int = 0,
            batch_size: int = 32,
            random_eps: bool = False,
            summary_writer: Union[str, bool, SummaryWriter] = False,
            verbose: bool = True,
    ):
        args = (estimator, norm, eps, eps_step, decay, max_iter,
                targeted, num_random_init, batch_size, random_eps,
                summary_writer, verbose)

        super().__init__(*args)
        self._attack = PGDNumpyConstr(*args)

    def vhost(self):
        """attach validation model to attack"""
        return self._attack
