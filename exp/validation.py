from typing import List, Dict, Tuple, Set

import numpy as np
from networkx import DiGraph, descendants

from exp.types import CONSTR_DICT


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
                 - a lambda function to evaluate target feature,
                    based on source feature values.
        """
        self.immutable = immutable or []
        self.constraints = constraints or {}
        self.dep_graph, self.desc = self.desc_graph(self.constraints)
        self.single_feat = dict(
            [(t, P) for (t, (s, P))
             in self.constraints.items() if (t,) == s])
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
            mask[:, target] *= mask_bits
            deps = list(self.desc[target])
            if deps and False in mask_bits:
                invalid = np.array((np.where(mask_bits == 0)[0]))
                mask[np.ix_(invalid, deps)] = 0

        # apply the constraints
        return adv * mask + ref * (1 - mask)

    @staticmethod
    def desc_graph(constraints: CONSTR_DICT) \
            -> Tuple[DiGraph, Dict[int, Set]]:
        """Construct a dependency graph from constraints.

        This allows to determine which target features
        are reachable from a source (omit self-loops).

        Arguments:
            constraints - constraints dictionary.

        Returns:
            The graph, a map of reachable nodes from each source.
        """
        g, targets = DiGraph(), list(constraints.keys())
        edges = [j for s in [[
            (src, tgt) for src in list(set(y)) if src != tgt]
            for tgt, (y, _) in constraints.items()] for j in s]
        nodes = list(set([s for s, _ in edges] + targets))
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        reachable = [(n, descendants(g, n)) for n in targets]
        return g, dict(reachable)

    def score(self, ref: np.ndarray, arr: np.ndarray) \
            -> Tuple[int, np.ndarray]:
        """Count number of valid instances.

        This metric should only be relevant is the constraints were
        not enforced during search, otherwise all should be valid.

        Arguments:
            ref - valid values.
            arr - modified records, to be evaluated.

        Returns:
            Total count of invalid records, array of invalid indices.
        """
        delta = np.subtract(arr.copy(), self.enforce(ref, arr))
        nonzr = (delta != 0).sum(1)
        return arr.shape[0] - np.count_nonzero(nonzr), nonzr


class Validatable:
    """Base class of a validatable attack."""
    v_model = None

    def set_validation(self, v: Validation):
        """Connect validation model."""
        self.v_model = v
