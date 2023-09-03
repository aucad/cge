from abc import ABC, abstractmethod
from typing import Dict, Callable, Tuple, Union

PREDICATE = Callable[[Tuple[float, ...]], bool]
"""Predicate is a function from R -> bool."""

CONSTR_DICT = Dict[int, Tuple[Tuple[int, ...], Union[PREDICATE, bool]]]
"""Constraints dictionary type.

   The key is the index of the target feature.
   The value is a tuple, containing:
   - a non-empty tuple of source feature indices.
   - a predicate (lambda function) to evaluate target feature
     validity, based on source feature values, or boolean False.
"""


class Validatable:
    """Base class for an attack with constraints"""
    v_model = None

    def vhost(self):
        """validation model owner"""
        return self


class Loggable(ABC):
    @abstractmethod
    def log(self):
        pass
