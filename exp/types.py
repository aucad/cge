from abc import ABC, abstractmethod
from typing import Dict, Callable, Tuple, Union, List

PREDICATE = \
    Union[Callable[[float], bool], Callable[[Tuple[float, ...]], bool]]
"""Predicate is a function from R -> bool."""

CONSTR_DICT = Dict[int, Tuple[Tuple[int, ...], Union[bool, PREDICATE]]]
"""Constraints dictionary type.

   The key is the index of the target feature.
   The value is a tuple, containing:
   - a non-empty tuple of source feature indices.
   - a predicate (lambda function) to evaluate target feature
     validity, based on source feature values.
"""

CONSTR_TXT = Union[Dict[int, Union[str, bool]],
                   Dict[int, Tuple[List[int], str]]]
"""Constraint type for experiment configuration files."""


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
