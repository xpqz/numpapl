"""
This is a Python port of the interesting bits of the APL 
parser given in the Dyalog dfns workspace (most likely due 
to John Scholes):

    https://dfns.dyalog.com/n_parse.htm

plus an interpreter atop NumPy. The parser implements a version
of the Bunda-Gerth algorithm, which is described here:

    https://dl.acm.org/doi/pdf/10.1145/384283.801081

See 
    https://dfns.dyalog.com/s_parse.htm for the actual grammar.

In a typical APL interpreter, the boundary between parsing and
interpretation is a bit blurry. The 'grammar' isn't context-free.
The reduction step in the Bunda-Gerth algorithm is the point 
where an expression can be evaluated, so there is no need to 
build an explicit AST (although we do!)

"""
from functools import reduce
import re
from string import ascii_letters, whitespace
from typing import Any, Callable, Optional, TypeAlias

import numpy as np

from environment import Env
from primitives import Voc

APLTYPE: TypeAlias = np.ndarray|int|float|complex|str

class APL:
    def __init__(self):
        self.functions = '+-×÷*=≥>≠∨∧⍒⍋⌽⍉⊖⍟⍱⍲!?∊⍴~↑↓⍳○*⌈⌊∇⍎⍕⊃⊂∩∪⊣⊢⊥⊤|≡≢,⍪⊆⌹'
        self.hybrids = '/⌿\⍀'
        self.monadic_operators = '⌸¨⍨'
        self.dyadic_operators = '⍣⌺@⍥⍤'
        
        # Grammar categories.
        # Note: the ordering is significant! The Bunda-Gerth binding tables 
        # below rely on this order.
        self.cats = [
            'A',   # 0: Arrays
            'F',   # 1: Functions
            'N',   # 2: Names (unassigned)
            'H',   # 3: Hybrid function/operators
            'AF',  # 4: Functions with curried left argument
            'JOT', # 5: Compose/null operand
            'DOT', # 6: Reference/product
            'DX',  # 7: Dotted ...
            'MOP', # 8: Monadic operators
            'DOP', # 9: Dyadic operators
            'IDX', # 10: Bracket indexing/axis specification
            'XAS', # 11: Indexed assignment [IDX]←
            'SL',  # 12: Subscript list ..;..;..
            'CLN', # 13: Colon token
            'GRD', # 14: Guard
            'XL',  # 15: Expression list ..⋄..⋄..
            'ARO', # 16: Assignment arrow ←
            'ASG', # 17: Name assignment
            'ERR'  # 18: Error
        ]

        # Lookup table for category indexing.
        self.ctab = dict([(c, i) for (i, c) in enumerate(self.cats)])

        # Brackets
        self.bkts = '()[]{}'
        self.bkt_pairs = ['()', '[]', '{}']
        self.lbs = ['(', '[', '{']
        self.rbs = [')', ']', '}']
        self.blabs = ['', 'IDX', 'F'] # Bracket labels: what's enclosed by each pair type?
        self.bcats = [19, 10, 1]      # Bracket label indices into self.cats
        
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

        # Bunda-Gerth result categories.
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

        # BEGIN{HACK} -- these should be transcribed into the BG tables proper.
        # Note: this means that some syntactically incorrect expressions won't be
        #       detected at parse time, but only at runtime.

        # Make the A node also behave like a Name instead of using the Name node.
        self.bmat[(0, 11)] = self.bmat[(2, 11)]
        self.bmat[(0, 16)] = self.bmat[(2, 16)]

        self.zmat[(0, 11)] = self.zmat[(2, 11)]
        self.zmat[(0, 16)] = self.zmat[(2, 16)]

        # Make the F node also behave like a Name instead of using the Name node.
        self.bmat[(1, 11)] = self.bmat[(2, 11)]
        self.bmat[(1, 16)] = self.bmat[(2, 16)]

        self.zmat[(1, 11)] = self.zmat[(2, 11)]
        self.zmat[(1, 16)] = self.zmat[(2, 16)]
        # END{HACK}

        # Extended bmat: pad with zeros in x and y
        self.xmat = np.pad(self.bmat, ((1, 1), (1, 1)), 'constant') 

    def tokenise(self, src: str) -> list[tuple]:
        """
        Tokeniser. Classify each atom as a tuple of (category, atom). Converts character
        vectors to numpy arraysm and numbers to proper numerics.
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
            elif ch == "⍬":
                pairs.append((0, ch))
            elif ch == '¯' or ch.isdigit() or ch == '.' and peek(src, i).isdigit():
                i, num = getnum(src, i)
                pairs.append((0, num))
                continue
            elif ch in ascii_letters+"⎕_": # A variable name
                i, name = getname(src, i)
                reftype = self.classify_name(name)   # BQN-style naming conventions
                pairs.append((reftype, name))
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
                pairs.append((self.ctab['SL'], ch))    # Subscript list separator
            elif ch == ':':
                pairs.append((self.ctab['CLN'], ch))   # Colon for guard
            elif ch == '⋄':
                pairs.append((self.ctab['XL'], ch))    # Expression list separator
            elif ch == '.':
                pairs.append((self.ctab['DOT'], ch))   # Dot: ref / product
            elif ch == '←':
                pairs.append((self.ctab['ARO'], ch))   # Assignment arrow
            elif ch in '⍺⍵#':
                pairs.append((0, ch))                  # Dfn arg arrays and root ns
            i += 1
        return pairs
    
    def bkt(self, bracket: str, t: tuple) -> tuple:
        """
        Classify node `t` based on it being enclosed by bracket-type
        `bracket`. This does the job of the following APL function:

        bkt ← {                       ⍝ bind of bracketed node [ ⍵ ].
            (cat expr)←⍵              ⍝ category of bracketed expr.
            zcat←(cat,1↓bcats)[lbs⍳⍺] ⍝ resulting category.
            zcat(⍺ expr)              ⍝ ⍺-bracketed node.
        }                             ⍝ :: left_bkt ∇ node → node
        """
        cat, expr = t
        # Note: in APL, `lbs⍳⍺` will not error if `⍺` isn't present, but instead 
        # default to 1+last index of `lbs`.
        try: 
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
    
    def classify_name(self, name: Any) -> int:
        if _name_is_array(name) or name in Env.system_arrs():
            return self.ctab['A']
        
        if _name_is_function(name) or name in Env.system_funs():
            return self.ctab['F']
        
        if _name_is_mop(name):
            return self.ctab['MOP']

        return self.ctab['DOP']
            
    def bind(self, stream: tuple) -> tuple:
        """
        Bunda-Gerth reduction. Find the first reducible token pair from `stream`, 
        right to left and evaluate the pair according to the grammar rules.
        """
        Aa, Bb, Cc = stream[1:4]
        D_, L = _unpack(stream[0])
        R, _D = _unpack(stream[4])

        if type(Aa) == int and Aa == Cc == -1: # Single node; we're done
            return Bb, # Note: trailling comma, as we're returning a tuple.
        
        if type(Aa) == int and Aa == Bb == -1: 
            # Partial parse; we can't reduce further. This is either a 
            # syntax error, or we've landed on a name that would need 
            # looking up.
            return stream
        
        # Separate the tuples into categories and tokens
        A, a = _unpack(Aa)
        B, b = _unpack(Bb)
        C, c = _unpack(Cc)

        if f'{b}{c}' in self.bkt_pairs[1:]: # Empty brackets [] {}
            return self.bind((rgt(((D_, L), Aa, self.ebk(b), R, _D)))) # rgt(∆_ L)Aa(ebk b)R _∆
        
        if f'{a}{c}' in self.bkt_pairs: # Bracketed single value Bb
            return self.bind(rgt((D_, L, self.bkt(a, Bb), R, _D)))  # (⊂a c)∊bkts:∇ rgt ∆_ L(a bkt Bb)R _∆
        
        if type(a) == str and a in self.rbs: # Right bracket: skip left.
            return self.bind(lft(lft(stream))) # (⊂a)∊rbs:∇ lft lft ⍵

        if self.xmat[(A+1, B+1)] >= self.xmat[(B+1, C+1)]: # A:B ≥ B:C → skip left.
            return self.bind(lft(stream)) 
        
        BbCc = (self.zmat[(B, C)], (b, c)) # BbCc←zmat[B;C],⊂b c ⍝ B bound with C.

        # BbCc = (self.zmat[(B, C)], (Bb, Cc)) # Retain node type info

        return self.bind(((D_, L), Aa, BbCc, R, _D)) # Binds with token to the right?
    
    def parse(self, pairs):
        """
        Parser entrypoint
        """
        eos = -1
        ll = [eos]+list(pairs[:-2])
        if ll == [-1]:
            D_ = -1
        else: # Left cons list: ∆_←↑{⍺ ⍵}⍨/⌽eos,¯2↓pairs
            D_ = tuple(reduce(lambda x, y: (x, y), ll)) # type: ignore 
        Aa, Bb, Cc = (eos, eos, *pairs, eos)[-3:] # 3-token window
        return self.bind((D_, Aa, Bb, Cc, eos))
    
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
    return (2+idx+len(data), np.array(list(data.replace("''", "'")), dtype='<U1'))

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
    return (idx+len(tok), val)

def getname(src: str, idx: int) -> tuple[int, str]:
    m = re.match(r'(^[⎕_a-z][_a-z0-9]*)', src[idx:], re.IGNORECASE)
    if not m:
        raise SyntaxError('SYNTAX ERROR: malformed name')
    data = m.group(1)
    if data[0] == '⎕':
        return (idx+len(data), data.upper())
    return (idx+len(data), data)

def _unpack(item: int|tuple) -> tuple:
    """
    Python doesn't do scalar extension, so we have to do it ourselves.
    """
    if type(item) == int:
        return (item, item)
    else:
        return item # type: ignore

def _name_is_array(name: Any) -> bool:
    return type(name) == str and name[0].islower()

def _name_is_function(name: Any) -> bool:
    return type(name) == str and name[0].isupper()

def _name_is_mop(name: Any) -> bool:
    return type(name) == str and len(name) > 1 and name[0] == '_' and name[1].isupper() and name[-1] != '_'

def _name_is_dop(name: Any) -> bool:
    return type(name) == str and len(name) > 1 and name[0] == '_' and name[1].isupper() and name[-1] == '_'

def rgt(stream: tuple) -> tuple:
    """
    Shift `stream` one step right

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
    Shift `stream` one step left

    lft ← {
        (∆_ L) A B C _∆ ← ⍵
        ∆_ L A B (C _∆)
    }
    """
    D_, L = _unpack(stream[0])
    A, B, C, _D = stream[1:]

    return D_, L, A, B, (C, _D)
        