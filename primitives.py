import itertools
import math
from string import ascii_letters
from typing import Callable, Optional

import numpy as np

import arr
from environment import Env
from errors import ArityError, DomainError, LengthError, NYIError, RankError

def nyi(*args):
    raise NotImplementedError('NYI')

def _is_int(a: np.ndarray) -> bool:
    return np.issubdtype(a.dtype, np.integer)

def _is_bool(a: np.ndarray) -> bool:
    return np.array_equal(a, a.astype(bool))

def iota(alpha: Optional[np.ndarray|str], omega: np.ndarray) -> np.ndarray:
    if alpha:
        raise NYIError('NYI ERROR: index-of (dyadic iota, a⍳b)')

    # Mondadic iota: index generator
    if not _is_int(omega):
        raise DomainError('DOMAIN ERROR: right arg must be integer-valued')
    
    if omega.ndim > 1:
        raise RankError("RANK ERROR: right arg rank must not be greater than 1")

    if omega.ndim == 0:
        return np.arange(int(omega))
    
    odo = [
        np.array(c)
        for c in itertools.product(*((range(c)) for c in omega))
    ]
    a = np.empty(len(odo), dtype=object)
    a[:] = odo
    return a.reshape(omega)

def tally(alpha: Optional[np.ndarray], omega: np.ndarray) -> np.ndarray:
    """
    Monadic: tally
    Dyadic: not-match
    """
    if alpha is None:
        if omega.ndim == 0:
            return np.array(1)
        return np.array(omega.shape[0])
    
    return match(alpha, omega)^1

def match(alpha: Optional[np.ndarray], omega: np.ndarray) -> np.ndarray:
    """
    TODO: check how this works on enclosed items.
    """
    if alpha is None:
        raise NotImplementedError('NYI: depth')
    
    if isinstance(alpha, np.ndarray) and isinstance(omega, np.ndarray):
        if alpha.shape != omega.shape:
            return np.array(0)
        return np.array(int(np.all([match(ai, bi) for ai, bi in zip(alpha.flat, omega.flat)])))
    else:
       return np.array(int(alpha == omega))
    
def rho(alpha: Optional[np.ndarray|str], omega: np.ndarray) -> np.ndarray:
    """
    Monadic ⍴: shape
    Dyadic ⍴:  reshape
    """
    if alpha is None:
        return np.array(list(omega.shape))
    
    assert isinstance(alpha, np.ndarray)
    bound = math.prod(alpha)

    if omega.ndim == 0:
        return np.repeat(omega, bound).reshape(alpha)
    
    # If we already have the right number of elements, we can 
    # just reshape the array == fast.
    if math.prod(omega.shape) == bound:
        return omega.reshape(alpha)

    # If we don't have the right number of elements, we need to
    # do some fancy itertools dance to repeat elements and/or
    # cut off a few.
    return np.array(list(itertools.islice(itertools.cycle(omega.ravel()), bound))).reshape(alpha)
    
def comma(alpha: Optional[np.ndarray], omega: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Mondadic: ravel
    Dyadic: catenate last (trailling axis)
    """
    if alpha is None:
        return np.ravel(omega)
    
    if alpha.ndim != omega.ndim:
        if omega.ndim == 0: # Scalar extension
            shape = list(alpha.shape)
            shape[-1] = 1
            return np.concatenate((alpha, omega.repeat(math.prod(shape)).reshape(shape)), axis=axis)
        raise LengthError('LENGTH ERROR')

    return np.concatenate((alpha, omega), axis=axis)

def right_shoe(alpha: Optional[np.ndarray], omega: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Monadic: first/disclose
    Dyadic: pick

    Note: APL's enclose and disclose fit poorly with numpy's array model
    """
    if alpha is None:
        if omega.ndim == 0:               # Disclose
            return arr.disclose(omega)
        return arr.disclose(omega[0])     # First

    if omega.ndim == 0:
        raise LengthError('LENGTH ERROR')

    if alpha > omega.shape[0]:
        raise LengthError('LENGTH ERROR')
    
    return arr.disclose(omega[alpha])     # Pick
    
def left_shoe(alpha: Optional[np.ndarray], omega: np.ndarray, axis: int = 0) -> np.ndarray:
    if axis != 0:
        raise NYIError("NYI enclose with axis != 0")
    if alpha is None:
        return arr.enclose(omega)
    raise NYIError

def rtack(alpha: Optional[np.ndarray], omega: np.ndarray) -> np.ndarray:
    return omega

def ltack(alpha: np.ndarray, omega: np.ndarray) -> np.ndarray:
    return alpha

def plus(alpha: Optional[np.ndarray], omega: np.ndarray) -> np.ndarray:
    if alpha is None:
        return np.conj(omega)
    return alpha + omega

def minus(alpha: Optional[np.ndarray], omega: np.ndarray) -> np.ndarray:
    if alpha is None:
        return -omega
    return alpha - omega

def times(alpha: Optional[np.ndarray], omega: np.ndarray) -> np.ndarray:
    if alpha is None:
        if omega == 0:
            return np.array(0)
        return omega / abs(omega)
    return alpha * omega

def divide(alpha: Optional[np.ndarray], omega: np.ndarray) -> np.ndarray:
    if alpha is None:
        return 1 / omega
    return alpha / omega
    
def replicate(alpha: Optional[np.ndarray], omega: np.ndarray) -> np.ndarray:
    """
    Replicate (trailling)
    """
    if alpha is None:
        raise SyntaxError('SYNTAX ERROR')

    return np.repeat(omega, alpha)

def replicate_first(alpha: Optional[np.ndarray], omega: np.ndarray, axis=None) -> np.ndarray:
    """
    Replicate (leading)
    """
    if alpha is None:
        raise SyntaxError('SYNTAX ERROR')

    return np.repeat(omega, alpha, axis=0)

def gets(alpha: str, omega: np.ndarray|Callable) -> Optional[np.ndarray]:
    Env.set(alpha, omega)
    if not callable(omega):
        return omega
    return None

def reduce(left: np.ndarray|Callable, right: Optional[np.ndarray|Callable], alpha: Optional[np.ndarray], omega: np.ndarray) -> np.ndarray:
    """
    Outward-facing '/' (trailling axis reduce)
    """
    if right is not None:
        raise ArityError("'/' takes no right operand")

    if alpha is not None:
        raise NYIError("left argument for function derived by '/' is not implemented yet")

    assert callable(left)
    return arr.foldr(omega, operand=left, axis=omega.ndim-1)

def reduce_first(left: np.ndarray|Callable, right: Optional[np.ndarray|Callable], alpha: Optional[np.ndarray], omega: np.ndarray) -> np.ndarray:
    """
    Outward-facing '⌿' (leading axis reduce)
    """
    if right is not None:
        raise ArityError("'⌿' takes no right operand")

    if alpha is not None:
        raise NYIError("left argument for function derived by '⌿' is not implemented yet")

    assert callable(left)
    return arr.foldr(omega, operand=left, axis=0)

def fun_gets(left: np.ndarray|Callable, right: Optional[np.ndarray|Callable], alpha: Optional[np.ndarray|str], omega: np.ndarray) -> np.ndarray:
    """
    a F← b
    """
    if right is not None:
        raise SyntaxError("SYNTAX ERROR: 'gets' takes no right operand")

    if left is None:
        raise SyntaxError("SYNTAX ERROR: 'gets' expects a left operand")

    if alpha is None:
        raise SyntaxError("SYNTAX ERROR: function derived by 'f←' takes a left argument")

    if not callable(left):
        raise SyntaxError("SYNTAX ERROR: 'f←' expects a function operand f")

    assert isinstance(omega, np.ndarray)
    assert type(alpha) == str

    return Env.amend(alpha, left, omega)

def each(left: np.ndarray|Callable, right: Optional[np.ndarray|Callable], alpha: Optional[np.ndarray], omega: np.ndarray) -> np.ndarray:
    """
    Each ¨ - monadic operator deriving monad
    """
    if right is not None:
        raise SyntaxError("SYNTAX ERROR: 'each' takes no right operand")

    if left is None:
        raise SyntaxError("SYNTAX ERROR: 'each' expects a left operand")

    if alpha is not None:
        raise SyntaxError("SYNTAX ERROR: function derived by 'each' takes no left argument")

    if not callable(left):
        raise SyntaxError("SYNTAX ERROR: 'each' expects a function operand")

    assert isinstance(omega, np.ndarray)

    return np.array([left(o) for o in omega]) # apply left operand to each element
    
def transpose(alpha: Optional[np.ndarray], omega: np.ndarray) -> np.ndarray:
    if alpha is None:
        return np.transpose(omega)
    
    # Note 1: NumPy's transpose does not work with repeated axes.
    # We can hardwire the simple case of all axes == 0 to give
    # us a diagonal.

    if alpha.ndim > 0 and np.all(alpha == alpha[0]):
        if alpha[0] != 0:
            raise NYIError('NYI ERROR: transpose with repeated axes ≠ 0')
        return np.diagonal(omega).copy()
    
    # Note: numpy's axis spec for dyadic transpose is NOT the same 
    # as APL's: we need the grade permutation of the axes.
    return np.transpose(omega, axes=np.argsort(alpha))

def grade_up(alpha: Optional[np.ndarray], omega: np.ndarray) -> np.ndarray:
    if alpha:
        raise NYIError('NYI ERROR: dyadic ⍋')
    return np.argsort(omega)

def grade_down(alpha: Optional[np.ndarray], omega: np.ndarray) -> np.ndarray:
    if alpha:
        raise NYIError('NYI ERROR: dyadic ⍒')
    return np.argsort(omega)[::-1]

def encode(alpha: Optional[np.ndarray], omega: np.ndarray) -> np.ndarray:
    if alpha is None:
        raise SyntaxError('SYNTAX ERROR: The function ⊤ requires a left argument')
    return np.array(np.unravel_index(omega, alpha))

def _decode(shape: np.ndarray, coords: np.ndarray) -> int:
    """
    Decode -- dyadic ⊥, aka `base`

    Evaluates `coords` in terms of the radix system defined by `shape`.
    
    Inverse of `encode()`

    https://aplwiki.com/wiki/Decode
    https://xpqz.github.io/cultivations/Decode.html

    """
    pos = 0
    rnk = len(shape)
    for axis in range(rnk):
        if axis >= len(coords):
            return pos
        pos += coords[axis]
        if axis != rnk - 1:
            pos *= shape[axis+1]
    return pos

def decode(alpha: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Decode - dyadic ⊥

    See https://aplwiki.com/wiki/Decode
        https://xpqz.github.io/cultivations/Decode.html
        https://help.dyalog.com/latest/index.htm#Language/Primitive%20Functions/Decode.htm

    2 ⊥ 1 1 0 1
  
    13

    24 60 60 ⊥ 2 46 40

    10000

    Note that we're really doing an inner product:

    (4 3⍴1 1 1 2 2 2 3 3 3 4 4 4)⊥3 8⍴0 0 0 0 1 1 1 1 0 0 1 1 0 0 1 1 0 1 0 1 0 1 0 1
    ┌→──────────────────┐
    ↓0 1 1 2  1  2  2  3│
    │0 1 2 3  4  5  6  7│
    │0 1 3 4  9 10 12 13│
    │0 1 4 5 16 17 20 21│
    └~──────────────────┘

    Dyalog's docs say:

        R←X⊥Y

        Y must be a simple numeric array.  X must be a simple numeric array.  R is the 
        numeric array which results from the evaluation of Y in the number system with radix X.

        X and Y are conformable if the length of the last axis of X is the same as the length 
        of the first axis of Y.  A scalar or 1-element vector is extended to a vector of the 
        required length.  If the last axis of X or the first axis of Y has a length of 1, the 
        array is extended along that axis to conform with the other argument.

        The shape of R is the catenation of the shape of X less the last dimension with the 
        shape of Y less the first dimension.  That is:

        ⍴R ←→ (¯1↓⍴X),1↓⍴Y

        For vector arguments, each element of X defines the ratio between the units for corresponding 
        pairs of elements in Y.  The first element of X has no effect on the result.

    """
    if alpha.ndim == 0: # Extend left scalar
        return np.array(np.ravel_multi_index(omega, alpha.repeat(len(omega)))) # type: ignore

    if omega.ndim == 0: # A right scalar might need to be extended, too
        return np.array(np.ravel_multi_index(omega.repeat(len(alpha), alpha))) # type: ignore
    
    # If the last axis of left or the first axis of right has a length of 1, this 
    # array is extended along that axis to conform with the other argument.
    if alpha.shape[-1] == 1:
        left = np.repeat(alpha, omega.shape[0], axis=-1)
    else:
        left = alpha

    if omega.shape[0] == 1:
        right = np.repeat(omega, alpha.shape[-1], axis=0)
    else:
        right = omega

    if left.shape[-1] != right.shape[0]:
        raise RankError('RANK ERROR')
    
    if left.ndim == right.ndim == 1:
        return np.array(np.ravel_multi_index(right, left)) # type: ignore
    
    # At least one side is higher-rank; we're doing an inner product
    shape = left.shape[:-1] + right.shape[1:]
    if left.ndim == 1: # Treat vector as 1-row matrix, wtf Dyalog
        left.reshape((1, left.shape[0]))

    ravel = []
    for lc in arr.major_cells(left):
        for rc in arr.major_cells(right.T):
            decoded = _decode(lc, rc)
            ravel.append(decoded)

    return np.array(ravel).reshape(shape)

def roll_deal(alpha: Optional[np.ndarray], omega: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng()

    if alpha is None: # Roll (monadic ?)
        if not _is_int(omega) or np.any(omega < 0):
            raise DomainError('DOMAIN ERROR: Roll right argument must consist of non-negative integer(s)')
        roll = rng.random(size=omega.shape)
        ints = (omega*roll).astype(int)
        return np.where(omega == 0, rng.random(), ints)
    
    # Deal (dyadic ?)

    # Y must be a simple scalar or 1-element vector containing a non-negative 
    # integer. X must be a simple scalar or 1-element vector containing a 
    # non-negative integer and X≤Y.
    if omega.ndim > 1 or omega.ndim == 1 and len(omega) != 1:
        raise LengthError('LENGTH ERROR: right arg must be non-negative integer scalar or singleton')
    if alpha.ndim > 1 or alpha.ndim == 1 and len(alpha) != 1:
        raise LengthError('LENGTH ERROR: left arg must be non-negative integer scalar or singleton')
    
    deal = omega if omega.ndim == 0 else omega[0]
    if deal < 0:
        raise DomainError('DOMAIN ERROR: Deal right argument must be non-negative')
    count = alpha if alpha.ndim == 0 else alpha[0]
    if count < 0:
        raise DomainError('DOMAIN ERROR: Deal left argument must be non-negative')
    if count > deal:
        raise DomainError('DOMAIN ERROR: Deal left argument must be less than or equal to right argument')

    # R is an integer vector obtained by making X random selections from ⍳Y 
    # without repetition. 
    return np.random.choice(deal, count, replace=False)

def tilde(alpha: Optional[np.ndarray], omega: np.ndarray) -> np.ndarray:
    if alpha is None:
        if not _is_int(omega) or not _is_bool(omega):
            raise DomainError('DOMAIN ERROR: right argument must be Boolean array')
        return omega ^ 1
    else:
        if alpha.ndim > 1: # Left must not exceed rank 1. Right will be ravelled.
            raise RankError('RANK ERROR: left argument must be a scalar or vector')
        return np.setdiff1d(alpha, np.ravel(omega), assume_unique=True)

def nyi(alpha: Optional[np.ndarray], omega: np.ndarray) -> np.ndarray: # type: ignore
    raise NYIError('NYI ERROR')

class Voc:
    """
    Voc is the global vocabulary of built-in arrays, functions and operators. This class should not
    be instantiated.
    """

    arrs: dict[str, np.ndarray] = {
        '⍬':   np.array([], dtype=int),
        '⎕IO': np.array(0),
        '⎕A':  np.array(ascii_letters[26:]),
        '⎕D':  np.arange(10)
    }

    funs: dict[str, Callable] = {
        '≢': lambda a, w: tally(Env.resolve(a), Env.resolve(w)),
        '≡': lambda a, w: match(Env.resolve(a), Env.resolve(w)),
        '⍳': lambda a, w: iota(Env.resolve(a), Env.resolve(w)),
        '⊂': lambda a, w: left_shoe(Env.resolve(a), Env.resolve(w)),
        '⊃': lambda a, w: right_shoe(Env.resolve(a), Env.resolve(w)),
        ',': lambda a, w: comma(Env.resolve(a), Env.resolve(w)),
        '⍴': lambda a, w: rho(Env.resolve(a), Env.resolve(w)),
        '⊢': lambda a, w: rtack(Env.resolve(a), Env.resolve(w)),
        '⊣': lambda a, w: ltack(Env.resolve(a), Env.resolve(w)),
        '+': lambda a, w: plus(Env.resolve(a), Env.resolve(w)),
        '-': lambda a, w: minus(Env.resolve(a), Env.resolve(w)),
        '×': lambda a, w: times(Env.resolve(a), Env.resolve(w)),
        '÷': lambda a, w: divide(Env.resolve(a), Env.resolve(w)),
        '⍉': lambda a, w: transpose(Env.resolve(a), Env.resolve(w)),
        '⍋': lambda a, w: grade_up(Env.resolve(a), Env.resolve(w)),
        '⍒': lambda a, w: grade_down(Env.resolve(a), Env.resolve(w)),
        '⊤': lambda a, w: encode(Env.resolve(a), Env.resolve(w)),
        '⊥': lambda a, w: decode(Env.resolve(a), Env.resolve(w)),
        '?': lambda a, w: roll_deal(Env.resolve(a), Env.resolve(w)),
        '~': lambda a, w: tilde(Env.resolve(a), Env.resolve(w)),
    }

    hybs: dict[str, tuple[Callable, Callable]] = {
        '/': (replicate,       reduce),
        '⌿': (replicate_first, reduce_first),
        '←': (gets,            fun_gets),
    }
    
    mops: dict[str, Callable] = {
        '¨': each,
        '⌸': nyi,
        '⍨': nyi
    }

    dops: dict[str, Callable] = {
        '⍤': nyi,
        '⍣': nyi,
        '⌺': nyi,
        '@': nyi,
        '⍥': nyi,
    }

    @classmethod
    def has_builtin(cls, f: str) -> bool:
        return f in cls.funs

    @classmethod
    def get_fn(cls, f: str) -> Callable:
        """
        Lookup a function from the global symbol table
        """
        try:
            return cls.funs[f]
        except KeyError:
            raise ValueError(f"VALUE ERROR: Undefined function: '{f}'")

    @classmethod
    def get_mop(cls, mop: str) -> Callable:
        try:
            return cls.mops[mop]
        except KeyError:
            raise ValueError(f"VALUE ERROR: Undefined monadic operator: '{mop}'")

    @classmethod
    def get_dop(cls, dop: str) -> Callable:
        try:
            return cls.dops[dop]
        except KeyError:
            raise ValueError(f"VALUE ERROR: Undefined dyadic operator: '{dop}'")

    @classmethod
    def get_hyb(cls, hyb: str) -> tuple[Callable, Callable]:
        try:
            return cls.hybs[hyb]
        except KeyError:
            raise ValueError(f"VALUE ERROR: Undefined hybrid: '{hyb}'")