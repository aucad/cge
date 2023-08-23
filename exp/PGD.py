from __future__ import absolute_import, division, print_function, \
    unicode_literals

import logging
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from art.attacks.evasion import ProjectedGradientDescent, \
    ProjectedGradientDescentNumpy
from art.summary_writer import SummaryWriter

from exp import Validatable

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE, OBJECT_DETECTOR_TYPE

logger = logging.getLogger(__name__)


class MyProjectedGradientDescentNumpy(
    ProjectedGradientDescentNumpy, Validatable
):

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
        x_ori = x.copy()
        batch = super()._compute(
            x, x_init, y, mask, eps, eps_step, project,
            random_init, batch_id_ext, decay, momentum)
        return self.v_model.enforce(x_ori, batch)


class PGDConst(ProjectedGradientDescent, Validatable):

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
        super().__init__(estimator)
        self._attack = MyProjectedGradientDescentNumpy(
            estimator=estimator,
            norm=norm,
            eps=eps,
            eps_step=eps_step,
            decay=decay,
            max_iter=max_iter,
            targeted=targeted,
            num_random_init=num_random_init,
            batch_size=batch_size,
            random_eps=random_eps,
            summary_writer=summary_writer,
            verbose=verbose,
        )
