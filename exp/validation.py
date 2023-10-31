from typing import Dict, Tuple

import numpy as np
from networkx import Graph, add_path, descendants, ancestors

from exp import CONSTR_DICT, categorize

ALL, DEP = 1, 2  # reset strategy


class Validation:
    """Constraint validation implementation."""

    def __init__(
            self, constraints: CONSTR_DICT,
            attr_range: list[Tuple[int, int]],
            mode=DEP
    ):
        """Initialize validation model

        Arguments:
            constraints - Constraints dictionary
            attr_range - ordered list of attribute ranges
            mode - reset strategy
        """
        self.constraints = constraints or {}
        self.scalars = attr_range
        self.immutable, self.mutable = x = categorize(self.constraints)
        self.deps = self.get_deps(*x)
        self.reset = mode if mode in [ALL, DEP] else DEP

    def enforce(self, ref: np.ndarray, adv: np.ndarray) -> np.ndarray:
        """Enforce feature constraints.

        Arguments:
            ref - reset point: must be known valid records.
            adv - adversarially perturbed records, potentially invalid.

        Returns:
            Valid adversarial records wrt. constraints.
        """
        vmap = np.ones(ref.shape, dtype=np.ubyte)
        vmap[:, self.immutable] = 0
        adv = adv * vmap + ref * (1 - vmap)

        vmap = np.ones(ref.shape, dtype=np.ubyte)
        for target, (sources, pred) in self.mutable.items():
            val_in = adv[:, sources]
            for i, ft_i in enumerate(sources):
                (mn, mx) = self.scalars[ft_i]
                val_in[:, i] = (val_in[:, i] * (mx - mn)) + mn
            bits = np.apply_along_axis(pred, 1, val_in)  # evaluate
            invalid = np.array((np.where(bits == 0)[0]))
            deps = range(ref.shape[1]) if self.reset == ALL else \
                self.deps[target]
            vmap[np.ix_(invalid, deps)] = 0  # apply
        return adv * vmap + ref * (1 - vmap)

    @staticmethod
    def get_deps(immutable, mutable) -> Dict[int, list]:
        """Construct a dependency lookup table to model constraints.

        This allows to determine which features are connected through
        constraints.

        Returns:
            A map of reachable features.
        """
        g, dep_nodes = Graph(), [s for (s, _) in mutable.values()]
        nodes = immutable + [c for dn in dep_nodes for c in dn]
        g.add_nodes_from(list(set(nodes)))
        for ngr in dep_nodes:
            add_path(g, ngr)
        r = [(k, list(({n} | ancestors(g, n) | descendants(g, n))))
             for k, n in [(k, s[0]) for k, (s, _) in mutable.items()]]
        return dict(r)
