"""
This is a python port of the interesting bits of the APL 
parser given in the Dyalog dfns workspace (most likely due 
to John Scholes):

    https://dfns.dyalog.com/n_parse.htm

See 
    https://dfns.dyalog.com/s_parse.htm for the actual grammar.

"""
from functools import reduce
import re
from string import ascii_letters, whitespace
from typing import Any, Callable, Optional, TypeAlias

import numpy as np

from environment import Env
from primitives import Voc

APLTYPE: TypeAlias = np.ndarray|int|float|complex|str

class APLParser:
    def __init__(self, parse_only: bool=False):
        self.parse_only = parse_only

        self.functions = '+-×÷*=≥>≠∨∧⍒⍋⌽⍉⊖⍟⍱⍲!?∊⍴~↑↓⍳○*⌈⌊∇⍎⍕⊃⊂∩∪⊥⊤|≡≢,⍪⊆⌹'
        self.hybrids = '/⌿\⍀'
        self.monadic_operators = '⌸¨⍨'
        self.dyadic_operators = '⍣⌺@⍥⍤'
        
        # Grammar categories
        # Note: order matters! The Bunda-Gerth binding tables below rely on
        # this order.
        self.cats = [
            'A', 'F', 'N', 'H', 'AF', 
            'JOT', 'DOT', 'DX', 'MOP', 
            'DOP', 'IDX', 'XAS', 'SL', 
            'CLN', 'GRD', 'XL', 'ARO', 
            'ASG', 'ERR'
        ]
        self.ctab = dict([(c, i) for (i, c) in enumerate(self.cats)])

        # Brackets
        self.bkts = '()[]{}'
        self.bkt_pairs = ['()', '[]', '{}']
        self.lbs = ['(', '[', '{']
        self.rbs = [')', ']', '}']
        self.blabs = ['', 'IDX', 'F'] # Bracket labels: what's enclosed by each pair type?
        self.bcats = [19, 10, 1]      # Bracket label indices in to self.cats
        
        # Bunda-Gerth binding strenghts.
        # The 9 at 0, 0 means that if a category 0 element (array) is bound
        # to a category 0 element (array) -- stranding, then the binding strength
        # is 9.
        self.bmat = np.array([
            [ 9,  7,  0,  7,  0,  0, 12,  0,  8,  0,  8,  0,  4,  2,  0,  1,  0,  0,  0],
            [ 6,  5,  0,  8,  5,  0,  0,  0,  8,  0,  8,  0,  0,  0,  0,  1,  0,  0,  0],
            [ 0,  0,  9,  0,  0,  0,  0,  0,  8,  0,  8, 10,  0,  0,  0,  1, 11,  0,  0],
            [ 0,  5,  0,  8,  0,  0,  0,  0,  8,  0,  8,  0,  0,  0,  0,  1,  0,  0,  0],
            [ 6,  5,  0,  0,  5,  0,  0,  0,  8,  0,  8,  0,  0,  0,  0,  1,  0,  0,  0],
            [ 8,  8,  8,  8,  8,  0,  0,  0,  8,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],
            [ 9,  8,  0,  8,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],
            [13, 13, 13, 13, 13, 13,  0,  0, 13, 13,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  8,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],
            [ 8,  8,  8,  8,  8,  8,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],
            [ 7,  7,  0,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 11,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 4,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  4,  0,  0,  0,  0,  0,  0],
            [ 2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],
            [ 2,  2,  2,  2,  2,  2,  0,  0,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 3,  3,  3,  3,  3,  3,  3,  0,  3,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        ])

        # Bunda-Gerth result categories
        # The 0 at 0, 0 means that if a category 0 element (array) is bound
        # to a category 0 element (array) -- stranding, then the resulting category
        # is 0 -- an array.
        self.zmat = np.array([
            [ 0,  4,  0,  4,  0,  0,  7,  0,  1,  0,  0,  0, 12, 14,  0, 15,  0,  0,  0],
            [ 0,  1,  0,  1,  1,  0,  0,  0,  1,  0,  1,  0,  0,  0,  0, 15,  0,  0,  0],
            [ 0,  0,  2,  0,  0,  0,  0,  0,  1,  0,  2, 17,  0,  0,  0, 15, 17,  0,  0],
            [ 0,  1,  0,  1,  0,  0,  0,  0,  1,  0,  3,  0,  0,  0,  0, 15,  0,  0,  0],
            [ 0,  1,  0,  0,  1,  0,  0,  0,  1,  0,  4,  0,  0,  0,  0, 15,  0,  0,  0],
            [ 8,  8,  8,  8,  8,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0, 15,  0,  0,  0],
            [18,  8,  0,  8,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 15,  0,  0,  0],
            [ 0,  1,  2,  3,  4,  5,  0,  0,  8,  9,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0, 18,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 15,  0,  0,  0],
            [ 8,  8,  8,  8,  8,  8,  0,  0,  0,  0,  0,  0,  0,  0,  0, 15,  0,  0,  0],
            [18, 18,  0, 18,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 11,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [12,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 12,  0,  0,  0,  0,  0,  0],
            [18,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 15,  0,  0,  0],
            [18, 18, 18, 18, 18, 18,  0,  0, 18, 18,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  1,  2,  3,  4,  5,  9,  0,  8,  9,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        ])

        self.xmat = np.pad(self.bmat, ((1, 1), (1, 1)), 'constant') # Extended bmat: pad with zeros in x and y

    def classify(self, src: str) -> list[tuple]:
        """
        Tokeniser. Classify each atom as a tuple of (category, atom).
        """
        i = 0
        pairs: list[tuple] = []
        while i<len(src):
            ch = src[i]
            if ch == '⍝':
                i = skip_comment(src, i)
                continue
            elif ch in whitespace:
                pass
            elif ch in self.bkts:
                pairs.append((-1, ch))
            elif ch == "'":
                i, s = getstring(src, i)
                pairs.append((0, s))
                continue
            elif ch == '¯' or ch.isdigit() or ch == '.' and peek(src, i).isdigit():
                i, num = getnum(src, i)
                pairs.append((0, num))
                continue
            elif ch in ascii_letters+"⎕_": # A variable name
                i, name = getname(src, i)
                pairs.append((self.ctab['N'], name))
                continue
            elif ch in self.functions:
                pairs.append((self.ctab['F'], ch))     # Primitive function
            elif ch in self.hybrids:
                pairs.append((self.ctab['H'], ch))     # Hybrid function/operator
            elif ch in self.monadic_operators:          
                pairs.append((self.ctab['MOP'], ch))   # Primitive monadic operator
            elif ch in self.dyadic_operators:
                pairs.append((self.ctab['DOP'], ch))   # Primitive dyadic operator
            elif ch == '∘':
                pairs.append((self.ctab['JOT'], ch))   # Jot: compose / null operand
            elif ch == ';':
                pairs.append((self.ctab['LST'], ch))   # Subscript list separator
            elif ch == ':':
                pairs.append((self.ctab['CLN'], ch))   # Expression list separator
            elif ch == '⋄':
                pairs.append((self.ctab['XL'], ch))    # Colon for guard
            elif ch == '.':
                pairs.append((self.ctab['DOT'], ch))   # Dot: ref / product
            elif ch == '←':
                pairs.append((self.ctab['ARO'], ch))   # Assignment arrow
            elif ch in '⍺⍵':
                pairs.append((0, ch))                        # Dfn arg arrays
            i += 1
        return pairs
    
    def bkt(self, bracket: str, t: tuple) -> tuple:
        """
        bkt ← {                       ⍝ bind of bracketed node [ ⍵ ].
            (cat expr)←⍵              ⍝ category of bracketed expr.
            zcat←(cat,1↓bcats)[lbs⍳⍺] ⍝ resulting category.
            zcat(⍺ expr)              ⍝ ⍺-bracketed node.
        }                             ⍝ :: left_bkt ∇ node → node
        """
        cat, expr = t
        try: # Note: in APL, lbs⍳⍺ will not error if ⍺ isn't present, but instead default to 1+last index of lbs
            zcat = ([cat]+self.bcats[1:])[self.lbs.index(bracket)]
        except ValueError:
            zcat = ([cat]+self.bcats[1:])[len(self.lbs)]

        return (zcat, (bracket, expr))
    
    def ebk(self, bracket: str) -> tuple:
        """
        Tag empty brackets with the category of the bracketed node:

            [] ⍝ Empty index list
            {} ⍝ Empty dfn
        """
        return (self.bcats[self.lbs.index(bracket)], bracket)
    
    def bind(self, stream: tuple) -> tuple:
        Aa, Bb, Cc = stream[1:4]
        D_, L = _unpack(stream[0])
        R, _D = _unpack(stream[4])

        if type(Aa) == int and Aa == Cc == -1: # Single node; we're done
            return Bb,
        
        if type(Aa) == int and Aa == Bb == -1: 
            # Partial parse; we can't reduce further. This is either a 
            # syntax error, or we've landed on a name that would need 
            # looking up. 
            return stream
        
        # Separate the tuples into categories and tokens
        A, a = _unpack(Aa)
        B, b = _unpack(Bb)
        C, c = _unpack(Cc)

        if type(a) == str and a in self.bkt_pairs[2:] and type(b) == str and b in self.bkt_pairs[2:]: # empty brackets []
            return self.bind((rgt((D_, L)), Aa, self.ebk(b), R, _D)) # rgt(∆_ L)Aa(ebk b)R _∆
        
        if type(a) == str and type(c) == str and a+c in self.bkt_pairs: # bracketed single value Bb
            return self.bind(rgt((D_, L, self.bkt(a, Bb), R, _D)))  # (⊂a c)∊bkts:∇ rgt ∆_ L(a bkt Bb)R _∆
        
        if a in self.rbs: # right bracket: skip left.
            return self.bind(lft(lft(stream))) # (⊂a)∊rbs:∇ lft lft ⍵

        if self.xmat[(A+1, B+1)] >= self.xmat[(B+1, C+1)]: # A:B ≥ B:C → skip left.
            return self.bind(lft(stream)) 

        if self.parse_only:
            BbCc = (self.zmat[(B, C)], (b, c)) # BbCc←zmat[B;C],⊂b c ⍝ B bound with C.
        else:
            BbCc = self.eval(B, C, b, c) # type: ignore
        return self.bind(((D_, L), Aa, BbCc, R, _D)) # binds with token to the right?
    
    def parse(self, src: str):
        pairs = self.classify(src)
        eos = -1
        ll = [eos]+pairs[:-2]
        if ll == [-1]:
            D_ = -1
        else: # Left cons list: ∆_←↑{⍺ ⍵}⍨/⌽eos,¯2↓pairs
            D_ = tuple(reduce(lambda x, y: (x, y), ll)) # type: ignore 
        Aa, Bb, Cc = (eos, eos, *pairs, eos)[-3:] # 3-token window
        return self.bind((D_, Aa, Bb, Cc, eos))
    
    def run(self, src: str) -> np.ndarray|Callable:
        tok = self.parse(src)
        if len(tok) == 1:
            if tok[0][0] == 0:
                if type(tok[0][1]) == np.ndarray:
                    return tok[0][1]
                elif type(tok[0][1]) == tuple and callable(tok[0][1][0]):
                    return tok[0][1][0](None, tok[0][1][1])
                
        raise NotImplementedError(f"Unrecognized token: {tok}")

    def array_(self, rcat: int, C: int, b: Any, c: Any) -> tuple:
        if C == self.ctab['A']:             # A:A -> 9 A  ⍝ Strand
            return (rcat, strand(b, c))
        
        if C == self.ctab['F']:             # A:F -> 7 AF ⍝ Left-side curried dyadic fun
            return (rcat, curry(b, Voc.get_fn(c)))
        
        if C == self.ctab['H']:             # A:H -> 7 AF ⍝ Left-side curried dyadic hybrid
            return (rcat, curry(b, Voc.get_hyb(c)[0])) # Function version of hybrid
        
        if C == self.ctab['DOT']:           # A:DOT -> 12 DX ⍝ Array to a dotted something
            raise NotImplementedError
        
        if C == self.ctab['MOP']:           # A:MOP -> 7 F ⍝ Derived fun from monadic op with array operand
            return (rcat, derive_function(Env.resolve(b), Voc.get_mop(c)))
        
        if C == self.ctab['IDX']:           # A:IDX -> 8 A ⍝ Array bracket index
            return (rcat, array_index(b, c[1]))
        
        if C == self.ctab['SL']:            # A:SL -> 4 SL ⍝ Semi-colon-sep subscript list
            return (rcat, subscript_list(b, c))
        
        if C == self.ctab['CLN']:           # A:CLN -> 2 GRD ⍝ Guard
            raise NotImplementedError
        
        if C == self.ctab['XL']:            # A:XL -> 1 XL ⍝ Diamond
            return (rcat, c) # Value of diamond list is value of last item
        
        raise NotImplementedError(f"Unknown category: {C}")
    
    def fun_(self, rcat: int, C: int, b: Any, c: Any) -> tuple:
        if C == self.ctab['A']:        # F:A -> 6 A  ⍝ Monadic function application
            return (rcat, apply(_fun_ref(b), c))
        
        if C == self.ctab['F']:        # F:F -> 8 F  ⍝ Derived function, atop
            return (rcat, atop(_fun_ref(b), _fun_ref(c)))
        
        if C == self.ctab['H']:        # F:H -> 8 F  ⍝ Mondadic hybrid operator with bound left function operand
            return (rcat, derive_function(_fun_ref(b), Voc.get_hyb(c)[1]))
        
        if C == self.ctab['AF']:       # F:AF -> 5 F  ⍝ Atop
            return (rcat, atop(_fun_ref(b), _fun_ref(c)))
        
        if C == self.ctab['MOP']:      # F:MOP -> 8 F  ⍝ Mondadic operator with bound left function operand
            return (rcat, derive_function(_fun_ref(b), Voc.get_mop(c)))
        
        if C == self.ctab['IDX']:      # F:IDX -> 8 F  ⍝ Bracket axis applied to function
            raise NotImplementedError('NYI ERROR: bracket axis')
        
        if C == self.ctab['XL']:       # F:XL -> 1 XL ⍝ Diamond
            return (rcat, c) # Value of diamond list is value of last item
        
        raise NotImplementedError(f"Unknown category: {C}")

    def name_(self, rcat: int, C: int, b: Any, c: Any) -> tuple:
        if C == self.ctab['N']:        # N:N -> 9 N  ⍝ Name stranding
            return (rcat, strand(b, c))
        
        if C == self.ctab['MOP']:      # N:MOP -> 8 F  ⍝ Mondadic operator with bound left name operand
            return (rcat, derive_function(_fun_ref(b), Voc.get_mop(c)))
        
        if C == self.ctab['IDX']:           # N:IDX -> 8 N Name bracket index/axis
            ref = Env.resolve(b)
            if type(ref) == np.ndarray:
                return (rcat, array_index(ref, c[1]))
            else:
                raise NotImplementedError('NYI ERROR: bracket axis')
            
        if C == self.ctab['XAS']:       # N:XAS -> 10 ASG  ⍝ Indexed assign: n[x]←
            raise NotImplementedError('NYI ERROR: indexed assign')
        
        if C == self.ctab['XL']:       # N:XL -> 1 XL ⍝ Diamond
            return (rcat, c) # Value of diamond list is value of last item
        
        if C == self.ctab['ARO']:       # N:ARO -> 11 ASG  ⍝ Assign: n←
            Env.set(b, c)
            return (rcat, c) # How do we do shy?
                    
        raise NotImplementedError(f"Unknown category: {C}")
    
    def eval(self, B: int, C: int, b: Any, c: Any) -> tuple:
        """
        Evaluate the expression b c
        """
        rcat = self.zmat[(B, C)]
        
        #---------- LEFT IS ARRAY -----------------
        if B == self.ctab['A']:
            return self.array_(rcat, C, b, c)

        #---------- LEFT IS FUNCTION --------------
        elif B == self.ctab['F']:
            return self.fun_(rcat, C, b, c)
        
        #---------- LEFT IS NAME - ----------------
        elif B == self.ctab['N']:
            return self.name_(rcat, C, b, c)
                
        return (rcat, (b, c))

def _simple_scalar(e: Any) -> bool:
    return isinstance(e, (int, float, complex, str))

def _payload(e: Any) -> APLTYPE:
    if type(e) == tuple:
        return e[1]
    return e

def _fun_ref(f: Any) -> Callable:
    if callable(f):
        return f
    return Voc.get_fn(f)
            
def strand(left: tuple|APLTYPE, right: tuple|APLTYPE) -> np.ndarray:
    # Flat strand of two simple scalars: 1 2
    if _simple_scalar(left) and _simple_scalar(right): 
        return np.hstack((left, right)) 
    
    # Flat strand of more than two simple scalars: 1 2 3
    if type(left) == np.ndarray and _simple_scalar(right):
        return np.hstack((left, right))
    
    if _simple_scalar(left) and type(right) == np.ndarray:
        return np.hstack((left, right))
    
    # Flat, bracketed, strand of two simple scalars: (1)(2)
    if (type(left) == tuple and type(right) == tuple and 
        _simple_scalar(left[1]) and _simple_scalar(right[1])):
        return np.hstack((left[1], right[1]))
    
    # Nested: (1 2 3)(4 5 6)
    nested = np.empty(2, dtype=object)
    nested[:] = [_payload(left), _payload(right)] # type: ignore
    return nested

def curry(left: np.ndarray, right: Callable) -> Callable:
    return lambda _, omega: right(left, omega)

def derive_function(left: np.ndarray|Callable, right: Callable) -> Callable:
    """
    Right is a monadic operator. Left is array or function. Bind the left operand, 
    and return a function

    Operators have the signature:

    def op(aa: np.ndarray|Callable, ww: Optional[np.ndarray|Callable], alpha: Optional[np.ndarray],  omega: np.ndarray)

    `derive_function` binds aa
    """
    return lambda alpha, omega: right(left, None, alpha, omega) # type: ignore

def array_index(left: np.ndarray, right: tuple|int) -> np.ndarray:
    """
    APL bracket index. Note: right is a subscript list or simple int scalar
    """
    return left[right] # maybe?

def subscript_list(left: tuple, right: tuple) -> tuple:
    """
    A subscript list is a tuple containing integers or other tuples.

        (1, (2, (3, 4)))

    each element representing selection crietria from a given axis.

    This function adds a new leading axis to the subscript list.
    """
    return left, *right

def apply(left: Callable, right: np.ndarray) -> np.ndarray:
    return left(None, right)

def atop(left: Callable, right: Callable) -> Callable:
    """
    -⍤÷ 4      ⍝ (  f⍤g y) ≡  f   g y
    ¯0.25

    12 -⍤÷ 4   ⍝ (x f⍤g y) ≡ (f x g y)
    ¯3
    """
    def derived(alpha: Optional[np.ndarray], omega: np.ndarray) -> np.ndarray:
        return left(None, right(alpha, omega))
    return derived

def rgt(stream: tuple) -> tuple:
    """
    Skip right

    rgt ← {
        ∆_ A B C (R _∆) ← ⍵
        (∆_ A) B C R _∆
    }
    """
    D_, A, B, C = stream[:-1]
    (R, _D) = _unpack(stream[-1])
    
    return (D_, A), B, C, R, _D

def lft(stream: tuple) -> tuple:
    """
    Skip left

    lft ← {
        (∆_ L) A B C _∆ ← ⍵
        ∆_ L A B (C _∆)
    }
    """
    D_, L = _unpack(stream[0])
    A, B, C, _D = stream[1:]

    return D_, L, A, B, (C, _D)
          
def _unpack(item: int|tuple) -> tuple:
    """
    Python doesn't do scalar extension, so we have to do it ourselves.
    """
    if type(item) == int:
        return (item, item)
    else:
        return item # type: ignore

def peek(src: str, idx: int) -> str:
    try:
        return src[idx+1]
    except IndexError:
        return ''
    
def skip_comment(src: str, i: int) -> int:
    while i < len(src) and src[i] != '\n':
        i += 1
    return i

def getstring(src: str, idx: int) -> tuple[int, np.ndarray]:
    m = re.match(r"'((?:''|[^'])*)'", src[idx:]) # Note: APL escapes single quotes with single quotes
    if not m:
        raise SyntaxError("SYNTAX ERROR: Unpaired quote")
    data = m.group(1)
    return (2+idx+len(data), np.array(list(data.replace("''", "'"))))

def get_cmplx(tok: str) -> complex:
    parts = tok.split('J')
    if len(parts) != 2:
        raise SyntaxError('SYNTAX ERROR: malformed complex scalar')
    re, im = parts
    try:
        cmplx = complex(float(re), float(im))
    except TypeError:
        raise SyntaxError('SYNTAX ERROR: malformed complex scalar')
    return cmplx

def getnum(src: str, idx: int) -> tuple[int, int|float|complex]:
    tok = ''
    start = idx
    m = re.match(r"[¯0-9eEjJ.]+", src[idx:])
    if not m:
        raise SyntaxError('SYNTAX ERROR: malformed numeric scalar')
    tok = m.group(0).upper().replace('¯', '-')
    if 'J' in tok:
        val = get_cmplx(tok)
    elif '.' in tok:
        val = float(tok)
    else:
        val = int(tok)
    # return (idx+len(tok), np.array(val))
    return (idx+len(tok), val)

def getname(src: str, idx: int) -> tuple[int, str]:
    m = re.match(r'(^[⎕_a-z][_a-z0-9]*)', src[idx:], re.IGNORECASE)
    if not m:
        raise SyntaxError('SYNTAX ERROR: malformed name')
    data = m.group(1)
    if data[0] == '⎕':
        return (idx+len(data), data.upper())
    return (idx+len(data), data)

if __name__ == "__main__":
    src = '{⍺+⍵}/⍳5'
    src = '(1 2 3 4)(5)'
    src = '1 2 3 4[1]'
    src = '1 2 3 4[1 0]'
    src = '+/1 2 3 4' # not yet
    src = '0 1 0 1/1 2 3 4'
    src = '-+'
    src = '5+7'
    src = '1 +⍨ 2'
    src = 'a (+⍤1 0) b'
    src = 'a[1] b[2] c[3]'
    src = '(a[1] b)[1]'
    p = APLParser(parse_only=True)
    ast = p.parse(src)
    print(ast)
