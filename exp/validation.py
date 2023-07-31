from typing import List, Dict, Callable, Tuple, Union

import numpy as np
from networkx import DiGraph, descendants

PREDICATE = Union[Callable[[float], bool], Callable[[List[float]], bool]]
"""Predicate is a function from R -> bool."""

CONSTR_DICT = Dict[int, Tuple[Tuple[int], PREDICATE]]
"""Constraints dictionary type."""


class Validation:
    """Constraint validation implementation."""

    def __init__(
            self,
            immutable: List[int] = None,
            constraints: CONSTR_DICT = None
    ):
        """Initialize validation module.

        Arguments:
            immutable - feature indices of immutable attributes.
                These are separate since they do not require evaluation.
            constraints - dictionary of enforceable predicates.
                The key is the index of the target feature.
                The value is a tuple, containing:
                 - a non-empty tuple of source feature indices
                 - a lambda function to evaluate validity of target feature.
        """
        self.immutable = immutable or []
        self.constraints = constraints or {}
        self.desc, self.dep_graph = self.desc_graph(self.constraints)
        self.single_feat = dict(
            [(k, P) for (k, (s, P))
             in self.constraints.items() if (k,) == s])
        self.multi_feat = dict(
            [x for x in self.constraints.items()
             if x[0] not in self.single_feat])

    def enforce(self, ref: np.ndarray, adv: np.ndarray) -> np.ndarray:
        """Enforce feature constraints.

        Arguments:
            ref - reset point (must be known valid records).
            adv - adversarially perturbed records (potentially invalid).

        Returns:
            Valid adversarial records wrt. constraints.
        """

        # initialize mask
        mask = np.ones(ref.shape, dtype=np.ubyte)

        # immutables are always 0
        for i in self.immutable:
            mask[:, i] = 0

        # evaluate single-feature constrains
        for index, pred in self.single_feat.items():
            inputs = adv[:, index]
            mask_bits = np.vectorize(pred)(inputs)  # evaluate
            mask[:, index] = mask_bits  # apply to mask

        # evaluate multi-variate constraints
        for target, (sources, pred) in self.multi_feat.items():
            inputs = adv[:, sources]
            mask_bits = np.apply_along_axis(pred, 1, inputs)
            mask[:, target] = mask_bits
            # propagate invalidity to dependents
            deps = list(self.desc[target])
            if deps and False in mask_bits:
                invalid = np.array((np.where(mask_bits == 0)[0]))
                mask[np.ix_(invalid, deps)] = 0

        # apply the constraints
        return adv * mask + ref * (1 - mask)

    @staticmethod
    def desc_graph(constraints: CONSTR_DICT):
        """Construct a dependency graph from constraints.

        This gives a graph to determine which target feature(s)
        are reachable from a source (omit self-loops).
        """
        g = DiGraph()
        targets = list(constraints.keys())
        edges = [j for s in [[
            (src, tgt) for src in list(set(y)) if src != tgt]
            for tgt, (y, _) in constraints.items()] for j in s]
        nodes = list(set([s for s, _ in edges] + targets))
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        return dict([(n, descendants(g, n)) for n in targets]), g

    def score_valid(self, ref: np.ndarray, arr: np.ndarray) \
            -> Tuple[int, np.ndarray]:
        """Count number of valid instances.

        This metric should only be relevant is the constraints were
        not enforced during search, otherwise all should be valid.
        """
        total = arr.shape[0]
        final_arr = arr.copy()
        correct = self.enforce(ref, arr)
        delta = np.subtract(final_arr, correct)
        nonzero = (delta != 0).sum(1)
        count_nz = np.count_nonzero(nonzero)
        return total - count_nz, nonzero


class Validatable:
    """Base class of a validatable attack."""
    v_model = None

    def set_validation(self, v: Validation):
        """Connect validation model."""
        self.v_model = v
