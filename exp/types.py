from abc import ABC, abstractmethod
from typing import Dict, Callable, Tuple, Union, List

PREDICATE = \
    Union[Callable[[float], bool], Callable[[Tuple[float, ...]], bool]]
"""Predicate is a function from R -> bool."""

CONSTR_DICT = Dict[int, Tuple[Tuple[int, ...], Union[bool, PREDICATE]]]
"""Constraints dictionary type."""

CONFIG_CONST_DICT = \
    Union[Dict[int, str], Dict[int, Tuple[List[int], str]]]
"""Constraint dictionary type for experiment configuration file."""


class Loggable(ABC):
    @abstractmethod
    def log(self):
        pass
