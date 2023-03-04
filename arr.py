
import math
from typing import Any, Callable, Generator

import numpy as np

from errors import RankError

def enclose(arr):
    if arr.shape == ():
        return arr
    return np.array([arr])

def _numeric(x: Any) -> bool:
    return isinstance(x, (int, float, complex))

def _string(x: Any) -> bool:
    return type(x) == str
    
def rank(arr: np.ndarray) -> int:
    return arr.ndim

def major_cells(arr: np.ndarray) -> Generator[np.ndarray, None, None]:
    """
    Yields the major cells of an array.
    """
    for i in range(arr.shape[0]):
        yield arr[i]
    
def catenate(a1: np.ndarray, a2: np.ndarray, axis=0) -> np.ndarray:
    """
    Concatenate two arrays along given existing axis, defaulting to first.
    """
    return np.concatenate((a1, a2), axis=axis)

def kcells(arr: np.ndarray, k: int) -> list[np.ndarray]:
    """
    Drop the first rank-k axes

    https://help.dyalog.com/latest/Content/Language/Introduction/Variables/Cells%20and%20Subarrays.htm
    https://mlochbaum.github.io/BQN/doc/array.html#cells

    Note: the resulting set of cells have a shape, too -- called `rsh` below. 
    When we implement ⍤, we'll need to use this, rather than just returning
    the cells as a list.
    """
    if k > rank(arr) or k < 0:
        raise RankError('RANK ERROR: kcells must be less than or equal to rank and greater than or equal to zero')

    if k == rank(arr):
        return enclose(arr.copy())

    # Shape and bound of result
    rsh = arr.shape[:rank(arr)-k]
    rbnd = math.prod(rsh)

    # Shape and bound of each cell
    csh = arr.shape[rank(arr)-k:]
    cbnd = math.prod(csh)
    data = np.ravel(arr) 
    return [
        np.array(data[cell*cbnd:(cell+1)*cbnd]).reshape(csh) 
        for cell in range(rbnd)
    ]

def foldr(arr: np.ndarray, operand: Callable, axis=0) -> np.ndarray:
    """
    Fold a function r-l over an array, returning the result as a new array.
    """
    f = np.frompyfunc(lambda x, y: operand(y, x), 2, 1)
    r = f.reduce(np.flip(arr, axis=axis), axis=axis)
    if type(r) == np.ndarray:
        return r
    return np.array(r)