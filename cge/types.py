from typing import Dict, Callable, Tuple, Union, Any

PREDICATE = Callable[[Tuple[float, ...]], bool]
"""Predicate is a function from seq R -> bool."""

CONSTR_DICT = Dict[Any, Tuple[Tuple[int, ...], Union[PREDICATE, bool]]]
"""Constraints dictionary type.

   * The key is a uid
   * The value is a pair of:
     - a non-empty tuple of source feature indices
     - a predicate (a function to evaluate feature validity
       based on source values), or a Boolean.
"""


class Validatable:
    """Base class for an attack with constraints"""
    cge = None

    def vhost(self):
        """validation model owner"""
        return self
