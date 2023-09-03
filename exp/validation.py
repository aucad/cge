from typing import Dict, Tuple, Set

import numpy as np
from networkx import DiGraph, descendants, ancestors

from exp import CONSTR_DICT, categorize

# reset strategy
ALL = 0
DEP = 1


class Validation:
    """Constraint validation implementation."""

    def __init__(self, constraints: CONSTR_DICT, attr_max: np.ndarray):
        self.constraints = constraints or {}
        self.scalars = attr_max
        self.dep_graph, self.desc = self.desc_graph(self.constraints)
        self.immutable, self.mutable = categorize(self.constraints)
        self.reset = DEP

    def enforce(self, ref: np.ndarray, adv: np.ndarray) -> np.ndarray:
        """Enforce feature constraints.

        Arguments:
            ref - reset point: must be known valid records.
            adv - adversarially perturbed records, potentially invalid.

        Returns:
            Valid adversarial records wrt. constraints.
        """
        vmap = np.ones(ref.shape, dtype=np.ubyte)
        for i in self.immutable:
            vmap[:, i] = 0
        adv = adv * vmap + ref * (1 - vmap)

        # mutable constraints
        vmap = np.ones(ref.shape, dtype=np.ubyte)
        nodes = range(ref.shape[1]) if self.reset == ALL else None
        for target, (sources, pred) in self.mutable.items():
            in_, sf = adv[:, sources], self.scalars[list(sources)]
            val_in = np.multiply(in_, sf)
            bits = np.apply_along_axis(pred, 1, val_in)  # evaluate
            invalid = np.array((np.where(bits == 0)[0]))
            deps = nodes or list(self.desc[target])
            vmap[:, target] *= bits
            vmap[np.ix_(invalid, deps)] = 0  # apply
        return adv * vmap + ref * (1 - vmap)

    @staticmethod
    def desc_graph(constraints: CONSTR_DICT) \
            -> Tuple[DiGraph, Dict[int, Set]]:
        """Construct a dependency graph to model constraints.

        This allows to determine which target nodes are reachable
        from each source node.

        Returns:
            The graph and a map of reachable nodes from each source.
        """
        g, targets = DiGraph(), list(constraints.keys())
        edges = [j for s in [[
            (src, tgt) for src in list(set(y)) if src != tgt]
            for tgt, (y, _) in constraints.items()] for j in s]
        nodes = list(set([s for s, _ in edges] + targets))
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        reachable = [(n, ancestors(g, n).union(descendants(g, n)))
                     for n in targets]
        return g, dict(reachable)
