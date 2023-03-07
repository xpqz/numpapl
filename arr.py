"""
Helper functions for array operations.
"""
import math
from typing import Any, Callable, Generator

import numpy as np

from errors import RankError

def enclose(arr):
    """
    This is doing hideous, suboptimal things to numpy arrays.

    APL's array model is boxed. NumPy's is not. So how to we make
    an enclosed object in NumPy? We can't, for example, do

        ⊂1 2 3 4       ⍝ Scalar
        ┌───────────┐
        │ ┌→──────┐ │
        │ │1 2 3 4│ │
        │ └~──────┘ │
        └∊──────────┘

        a = np.array(np.array([1, 2, 3, 4]), dtype=object).reshape(()) # Error
    """
    if arr.shape == ():
        return arr
    
    c = np.empty(1, dtype=object)
    c[:] = [arr]
    return c.reshape(())

def disclose(arr):
    if type(arr) == np.ndarray:
        if arr.ndim == 0:               # Disclose
            if arr.dtype != object:     # Simple scalar
                return arr 
            return arr.reshape((1,))[0] # Enclosed object
    return arr
        
def major_cells(arr: np.ndarray) -> Generator[np.ndarray, None, None]:
    """
    Yields the major cells of an array.
    """
    for i in range(arr.shape[0]):
        yield arr[i]

def kcells(arr: np.ndarray, k: int) -> list[np.ndarray]:
    """
    Drop the first rank-k axes

    https://help.dyalog.com/latest/Content/Language/Introduction/Variables/Cells%20and%20Subarrays.htm
    https://mlochbaum.github.io/BQN/doc/array.html#cells

    Note: the resulting set of cells have a shape, too -- called `rsh` below. 
    When we implement ⍤, we'll need to use this, rather than just returning
    the cells as a list.
    """
    if k > arr.ndim or k < 0:
        raise RankError('RANK ERROR: kcells must be less than or equal to rank and greater than or equal to zero')

    if k == arr.ndim:
        return enclose(arr.copy())

    # Shape and bound of result
    rsh = arr.shape[:arr.ndim-k]
    rbnd = math.prod(rsh)

    # Shape and bound of each cell
    csh = arr.shape[arr.ndim-k:]
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