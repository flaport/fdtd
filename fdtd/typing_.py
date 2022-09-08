""" common types """

## Standard Library Types
from typing import Union, Tuple, List
from numbers import Number

## Numpy types
from numpy import ndarray

## Torch types
try:
    from torch import Tensor
except ImportError:
    Tensor = None

## Backend types
if Tensor is not None:
    Tensorlike = Union[ndarray, Tensor]
else:
    Tensorlike = ndarray

## Composite types
ListOrSlice = Union[List[int], slice]
IntOrSlice = Union[int, slice]
