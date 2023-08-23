# flake8: noqa: E501

import logging

import numpy as np
from art.attacks.evasion import ZooAttack

from exp import Validatable

logger = logging.getLogger(__name__)


class ZooConst(ZooAttack, Validatable):

    def _generate_batch(self, x_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
        """
        Run the attack on a batch of images and labels.

        :param x_batch: A batch of original examples.
        :param y_batch: A batch of targets (0-1 hot).
        :return: A batch of adversarial examples.
        """
        # Initialize binary search
        c_current = self.initial_const * np.ones(x_batch.shape[0])
        c_lower_bound = np.zeros(x_batch.shape[0])
        c_upper_bound = 1e10 * np.ones(x_batch.shape[0])

        # Initialize best distortions and best attacks globally
        o_best_dist = np.inf * np.ones(x_batch.shape[0])
        o_best_attack = x_batch.copy()

        # Start with a binary search
        for bss in range(self.binary_search_steps):
            logger.debug(
                "Binary search step %i out of %i (c_mean==%f)",
                bss,
                self.binary_search_steps,
                np.mean(c_current),
            )

            # Run with 1 specific binary search step
            best_dist, best_label, best_attack = self._generate_bss(x_batch, y_batch, c_current)
            best_attack = self.v_model.enforce(x_batch, best_attack)

            # Update best results so far
            o_best_attack[best_dist < o_best_dist] = best_attack[best_dist < o_best_dist]
            o_best_dist[best_dist < o_best_dist] = best_dist[best_dist < o_best_dist]

            # Adjust the constant as needed
            c_current, c_lower_bound, c_upper_bound = self._update_const(
                y_batch, best_label, c_current, c_lower_bound, c_upper_bound
            )

        return o_best_attack
