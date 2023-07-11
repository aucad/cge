import numpy as np
from typing import List, Dict, Callable, Tuple

PREDICATE = Callable[[float], bool]
"""Predicate is a function from R -> bool."""


class Validation:
    """Constraint validation implementation"""

    def __init__(
            self,
            immutable: List[int] = None,
            constraints: Dict[int, PREDICATE] = None
    ):
        """Initialize validation module.

        Arguments:
            immutable - feature indices of immutable attributes.
                These are separate since they don't require evaluation.
            constraints - collection of enforceable predicates.
                The key is the feature index.
                The value is a lambda function R -> bool.
                (Not sure about multivariate yet!)
        """
        self.immutable = immutable or []
        self.constraints = constraints or {}

    def enforce(self, ref: np.ndarray, adv: np.ndarray) -> np.ndarray:
        """Enforce feature constraints.

        Arguments:
            ref - reset point (valid).
            adv - adversarially perturbed records (potentially invalid).

        Returns:
            Valid adversarial records wrt. constraints.
        """

        # initialize mask as all 1s
        mask = np.ones(ref.shape, dtype=np.ubyte)

        # immutables are always 0
        for i in self.immutable:
            mask[:, i] = 0

        # iterate the evaluable constraints
        for index, f in self.constraints.items():
            input_values = adv[:, index]  # column vector
            mask_bits = np.vectorize(f)(input_values)  # evaluate
            mask[:, index] = mask_bits  # apply to mask
            # TODO: multivariate, maybe tuple of indices as a key?

        # enforce the constraints
        result = adv * mask + ref * (1 - mask)

        return result

    def score_valid(self, ref: np.ndarray, arr: np.ndarray)\
            -> Tuple[int, np.ndarray]:
        """Count number of valid instances."""
        total = arr.shape[0]
        delta = np.subtract(arr, self.enforce(ref, arr))
        nonzero = (delta != 0).sum(1)
        count_nz = np.count_nonzero(nonzero)
        return total - count_nz, nonzero


class Validatable:
    """Base class of a validatable entity."""
    v_model = None

    def set_validation(self, v: Validation):
        """connect validation model"""
        self.v_model = v
