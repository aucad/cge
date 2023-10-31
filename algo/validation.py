from typing import Dict, Tuple, List, Any

import numpy as np

from . import CONSTR_DICT

ALL, DEP = 1, 2  # reset strategy


class Validation:
    """Constraint validation implementation."""

    def __init__(
            self, constraints: CONSTR_DICT,
            attr_range: List[Tuple[int, int]],
            mode: int = DEP
    ):
        """Initialize validation model.

        Arguments:
            constraints - Constraints dictionary
            attr_range - ordered list of attribute ranges
            mode - reset strategy
        """
        self.constraints = constraints or {}
        self.immutable, self.mutable = \
            Validation.categorize(self.constraints)
        self.deps = self.dep_map(self.mutable)
        self.scalars = attr_range
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
    def dep_map(mutable: CONSTR_DICT) -> Dict[int, List]:
        """Construct a dependency lookup table for mutable constraints.

        This allows to determine which features are connected through
        constraints.

        Arguments:
            mutable - Mutable constraints dictionary

        Returns:
            A map of reachable features.
        """
        deps, result = [set(s) for (s, _) in mutable.values()], {}
        while deps:
            x = deps.pop(0)
            if y := next((y for y in deps if x & y), None):
                deps.pop(deps.index(y))
                deps.append(x | y)
            else:
                result.update(dict([
                    (k, sorted(list(x))) for k, (s, _)
                    in mutable.items() if x & set(s)]))
        return result

    @staticmethod
    def categorize(cd: CONSTR_DICT) -> Tuple[List[Any], CONSTR_DICT]:
        """Categorize constraints by kind.

        Arguments:
            cd - Constraints dictionary

        Returns:
            A tuple of constrains where (immutable, mutable).
        """
        immutable = [k for k, (_, P) in cd.items() if P is False]
        mutable = dict([x for x in cd.items() if x[0] not in immutable])
        return immutable, mutable
