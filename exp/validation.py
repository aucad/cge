from typing import List, Dict, Callable, Tuple

import numpy as np
from networkx import DiGraph, descendants

PREDICATE = Callable[[float], bool]
"""Predicate is a function from R -> bool."""

CONSTR_DICT = Dict[int, Tuple[Tuple[int], PREDICATE]]
"""Constraints dictionary type."""


class Validation:
    """Constraint validation implementation"""

    def __init__(
            self,
            immutable: List[int] = None,
            constraints: CONSTR_DICT = None
    ):
        """Initialize validation module.

        Arguments:
            immutable - feature indices of immutable attributes.
                These are separate since they don't require evaluation.
            constraints - collection of enforceable predicates.
                The key is the feature index.
                The value is a tuple of:
                 - tuple of source indices
                 - lambda function float -> bool.
        """
        self.immutable = immutable or []
        self.constraints = constraints or {}
        self.desc = self.desc_graph(self.constraints)
        for v in self.desc.items():
            print(v)

    @staticmethod
    def desc_graph(constraints: CONSTR_DICT):
        g = DiGraph()
        targets = list(constraints.keys())
        edges = [j for s in [[
            (src, tgt) for src in list(set(y)) if src != tgt]
            for tgt, (y, _) in constraints.items()] for j in s]
        nodes = list(set([s for s, _ in edges] + targets))
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        return dict([(n, descendants(g, n)) for n in targets])

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
        for index, (src, f) in self.constraints.items():
            input_values = adv[:, index]  # column vector
            mask_bits = np.vectorize(f)(input_values)  # evaluate
            mask[:, index] = mask_bits  # apply to mask
            # TODO: apply multivariate case

        # enforce the constraints
        result = adv * mask + ref * (1 - mask)

        return result

    def score_valid(self, ref: np.ndarray, arr: np.ndarray) \
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
