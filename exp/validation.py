import numpy as np
from typing import List, Dict, Callable

PREDICATE = Callable[[float], bool]
"""Predicate is a function from R -> bool."""


class Validation:
    """Constraint validation implementation"""

    def __init__(
            self,
            original: np.array = None,
            immutable: List[int] = None,
            constraints: Dict[int, PREDICATE] = None,
            apply=True
    ):
        """Initial setup.

        Arguments:
            original - valid data records
            immutable - feature indices of immutable attributes.
                These are separate because they don't require evaluation.
            constraints - collection of enforceable predicates.
                The key is the feature index.
                The value is a lambda function R -> bool.
                (Not sure about multivariate yet!)
        """
        self.original = np.copy(original) if \
            original is not None else np.array([])
        self.immutable = immutable or []
        self.constraints = constraints or {}
        self.disabled = not apply

    def reset(self, ori: np.array):
        """reset comparison data"""
        self.original = ori

    @property
    def __has_constraints(self):
        """Check if some constraints have been specified."""
        return len(self.immutable) + len(self.constraints.keys()) > 0

    def __enforce(self, adv: np.array) -> np.array:

        # initialize mask as all 1s
        mask = np.ones(self.original.shape, dtype=np.ubyte)

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
        result = adv * mask + self.original * (1 - mask)

        return result

    def enforce(self, adv: np.array) -> np.array:
        """Enforce feature constraints.

        Arguments:
            adv - adversarially perturbed records (potentially invalid).

        Returns:
            Valid adversarial records, enforcing the provided constraints.
        """
        return adv if (self.disabled or not self.__has_constraints) else \
            self.__enforce(adv)

    def score_valid(self, arr: np.array):
        """For some adversarial sample, calculate percent valid 0.0-1.0"""
        total = arr.shape[0]
        validated = self.__enforce(arr)
        delta = self.original - validated
        temp = (delta != 0).sum(1)
        nonzero = np.count_nonzero(temp)
        return (total - nonzero) / total

    @staticmethod
    def parse_pred(config: dict):
        """Parse text input of a constraint predicate.
        (There maybe some better approach not using eval).
        """
        result = {}
        for key, value in config.items():
            result[key] = eval(value)
        return result
