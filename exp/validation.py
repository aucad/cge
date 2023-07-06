import numpy as np
from typing import List, Dict, Callable

PREDICATE = Callable[[float], bool]
"""Predicate is a function from R -> bool."""


class Validator:
    """Constraint validation implementation"""

    def __init__(
            self,
            original: np.array,
            immutable: List[int] = None,
            constraints: Dict[int, PREDICATE] = None
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
        self.original = original
        self.immutable = immutable or []
        self.constraints = constraints or {}

    @property
    def has_constraints(self):
        """Check if some constraints have been specified."""
        return len(self.immutable) + len(self.constraints.keys()) > 0

    def enforce(self, adv: np.array) -> np.array:
        """Enforce feature constraints.

        Arguments:
            adv - adversarially perturbed records (potentially invalid).

        Returns:
            Valid adversarial records, enforcing the provided constraints.
        """
        if not self.has_constraints:
            return adv

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

        # we can update original here, if we want
        # self.original = np.copy(result)

        return result
