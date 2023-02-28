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
from typing import Any

import numpy as np

class APLParser:
    def __init__(self):
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
        
        # Bracket pairs
        self.bkts = [
            '(', ')', 
            '[', ']', 
            '{', '}'
        ]
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
            elif ch in ascii_letters+"⎕_": # Named arrays, functions or operators
                i, name = getname(src, i)
                pairs.append((self.cats.index('N'), name))
                continue
            elif ch in self.functions:
                pairs.append((self.cats.index('F'), ch))     # Primitive function
            elif ch in self.hybrids:
                pairs.append((self.cats.index('H'), ch))     # Hybrid function/operator
            elif ch in self.monadic_operators:          
                pairs.append((self.cats.index('MOP'), ch))   # Primitive monadic operator
            elif ch in self.dyadic_operators:
                pairs.append((self.cats.index('DOP'), ch))   # Primitive dyadic operator
            elif ch == '∘':
                pairs.append((self.cats.index('JOT'), ch))   # Jot: compose / null operand
            elif ch == ';':
                pairs.append((self.cats.index('LST'), ch))   # Subscript list separator
            elif ch == ':':
                pairs.append((self.cats.index('CLN'), ch))   # Expression list separator
            elif ch == '⋄':
                pairs.append((self.cats.index('XL'), ch))    # Colon for guard
            elif ch == '.':
                pairs.append((self.cats.index('DOT'), ch))   # Dot: ref / product
            elif ch == '←':
                pairs.append((self.cats.index('ARO'), ch))   # Assignment arrow
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
        zcat = ([cat]+self.bcats[1:])[self.lbs.index(bracket)]

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

        if a in self.bkts[2:] and b in self.bkts[2:]: # empty brackets []
            return self.bind((rgt((D_, L)), Aa, self.ebk(b), R, _D)) # rgt(∆_ L)Aa(ebk b)R _∆
        
        if a in self.bkts and c in self.bkts: # bracketed single value Bb
            return self.bind((D_, L, self.bkt(a, Bb), R, _D))  # (⊂a c)∊bkts:∇ rgt ∆_ L(a bkt Bb)R _∆
        
        if a in self.rbs: # right bracket: skip left.
            return self.bind(lft(lft(stream))) # (⊂a)∊rbs:∇ lft lft ⍵

        if self.xmat[(A+1, B+1)] >= self.xmat[(B+1, C+1)]: # A:B ≥ B:C → skip left.
            return self.bind(lft(stream)) 

        BbCc = (self.zmat[(B, C)], (b, c)) # BbCc←zmat[B;C],⊂b c ⍝ B bound with C. This is the eval hook
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
    
    def eval(self, B: int, C: int, b: Any, c: Any) -> tuple:
        pass

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

def getstring(src: str, idx: int) -> tuple[int, str]:
    m = re.match(r"'((?:''|[^'])*)'", src[idx:]) # Note: APL escapes single quotes with single quotes
    if not m:
        raise SyntaxError("SYNTAX ERROR: Unpaired quote")
    data = m.group(1)
    return (2+idx+len(data), data.replace("''", "'"))

def get_cmplx(tok: str) -> complex:
    parts = tok.split('J')
    if len(parts) != 2:
        raise SyntaxError('SYNTAX ERROR: malformed complex scalar')
    (re, im) = parts
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
    tok = m.group(0).upper()
    tok = tok.replace('¯', '-')
    if 'J' in tok:
        return (idx+len(tok), get_cmplx(tok))
    if '.' in tok:
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

if __name__ == "__main__":
    src = '{⍺+⍵}/⍳5'
    p = APLParser()
    ast = p.parse(src)
    print(ast)