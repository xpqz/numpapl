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
build an explicit AST.

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
    def __init__(self, parse_only: bool=False):
        self.parse_only = parse_only

        self.functions = '+-×÷*=≥>≠∨∧⍒⍋⌽⍉⊖⍟⍱⍲!?∊⍴~↑↓⍳○*⌈⌊∇⍎⍕⊃⊂∩∪⊣⊢⊥⊤|≡≢,⍪⊆⌹'
        self.hybrids = '/⌿\⍀'
        self.monadic_operators = '⌸¨⍨'
        self.dyadic_operators = '⍣⌺@⍥⍤'
        
        # Grammar categories.
        # Note: the ordering is significant! The Bunda-Gerth binding tables 
        # below rely on this order.
        self.cats = [
            'A',   # Arrays
            'F',   # Functions
            'N',   # Names (unassigned)
            'H',   # Hybrid function/operators
            'AF',  # Functions with curried left argument
            'JOT', # Compose/null operand
            'DOT', # Reference/product
            'DX',  # Dotted ...
            'MOP', # Monadic operators
            'DOP', # Dyadic operators
            'IDX', # Bracket indexing/axis specification
            'XAS', # Indexed assignment [IDX]←
            'SL',  # Subscript list ..;..;..
            'CLN', # Colon token
            'GRD', # Guard
            'XL',  # Expression list ..⋄..⋄..
            'ARO', # Assignment arrow ←
            'ASG', # Name assignment
            'ERR'  # Error
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

        # Extended bmat: pad with zeros in x and y
        self.xmat = np.pad(self.bmat, ((1, 1), (1, 1)), 'constant') 

    def classify(self, src: str) -> list[tuple]:
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
                # reftype = self.classify_name(name)   # BQN-style naming conventions
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

        # if self.parse_only:
        return (zcat, (bracket, expr))
        # else:
        #     return zcat, expr
    
    def ebk(self, bracket: str) -> tuple:
        """
        Tag empty brackets with the category of the bracketed node:

            [] ⍝ Empty index list
            {} ⍝ Empty dfn
        """
        return (self.bcats[self.lbs.index(bracket)], bracket)
    
    def classify_name(self, name: Any) -> int:
        if _name_is_array(name):
            return self.ctab['A']
        
        if _name_is_function(name):
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
            # return R, # TODO: expressions? syntax errors?
            # return self.parse2(_flat(stream[-1], [])) # TODO: this should not be needed :/
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

        # BbCc←zmat[B;C],⊂b c ⍝ B bound with C.
        if self.parse_only:
            BbCc = (self.zmat[(B, C)], (b, c)) 
        else:
            BbCc = self.eval(B, C, b, c) # type: ignore
        return self.bind(((D_, L), Aa, BbCc, R, _D)) # Binds with token to the right?
    
    def parse(self, src: str):
        """
        Parser entrypoint
        """
        pairs = self.classify(src)
        eos = -1
        ll = [eos]+pairs[:-2]
        if ll == [-1]:
            D_ = -1
        else: # Left cons list: ∆_←↑{⍺ ⍵}⍨/⌽eos,¯2↓pairs
            D_ = tuple(reduce(lambda x, y: (x, y), ll)) # type: ignore 
        Aa, Bb, Cc = (eos, eos, *pairs, eos)[-3:] # 3-token window
        return self.bind((D_, Aa, Bb, Cc, eos))
    
    def parse2(self, pairs):
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
    
    def run(self, src: str) -> np.ndarray|Callable:
        """
        Interpreter entrypoint
        """
        tok = self.parse(src)
        if len(tok) == 1:
            if tok[0][0] == 0:   # Array
                if type(tok[0][1]) == tuple and callable(tok[0][1][0]):
                    return tok[0][1][0](None, tok[0][1][1])
                else:
                    return Env.resolve(tok[0][1])
            elif tok[0][0] == 2: # Name
                return Env.resolve(tok[0][1])
                
        # If we have more than one token left, we have a set of 
        # expressions separated by diamonds.
          
        raise NotImplementedError(f"Parse error: {tok}")

    def A_(self, rcat: int, C: int, b: Any, c: Any) -> tuple:
        """
        Left node is array.
        """
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
            return (rcat, (b, c))
        
        raise NotImplementedError(f"Unknown category: {C}")
    
    def F_(self, rcat: int, C: int, b: Any, c: Any) -> tuple:
        """
        Left node is function.
        """
        if C == self.ctab['A']:        # F:A -> 6 A  ⍝ Monadic function application
            return (rcat, apply(_fun_ref(b), c))
        
        if C == self.ctab['F']:        # F:F -> 5 F  ⍝ Derived function, atop
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
            return (rcat, (b, c)) # Value of diamond list is value of last item
        
        raise NotImplementedError(f"Unknown category: {C}")

    def N_(self, rcat: int, C: int, b: Any, c: Any) -> tuple:
        """
        Left node is name, which is a reference to either an array (a←4 5 5) or something
        callable (e.g. a←+/).

        We've made the decision to enforce BQN's variable naming conventions to be able to 
        resolve this at parse time. 

            1. Array references must start with a lower-case letter:

                a←4 5 5                     ⍝ An array reference
                hello_world ← 'hello world' ⍝ An array reference 

            2. Function references start with an upper-case letter:

                Sum ← +/                    ⍝ A function reference    
                Sum2 ← {+/⍵}                ⍝ A function reference

            3. References to monadic operators must start with an underscore, followed by
               an upper-case letter:

                _Mop ← {...⍺⍺...}           ⍝ A monadic operator reference

            3. References to dyadic operators must start with an underscore, followed by
               an upper-case letter, and end with an underscore:

                _Dop_ ← {...⍺⍺..⍵⍵...}      ⍝ A dyadic operator reference 

        """
        # The assignment is the only one we must delay the 
        # resolving of the left node for...
        if C == self.ctab['ARO']:            # N:ARO -> 11 ASG  ⍝ Assign: n←
            return (rcat, curried_assign(b))
        
        # ...whereas for all the others we should be able to
        # resolve the left node.
        # reftype = self.classify_name(b)
        # ref = Env.resolve(b)
        # if reftype == self.ctab['A']:
        #     return self.A_(reftype, C, ref, c)
        
        # if reftype == self.ctab['F']:
        #     return self.F_(reftype, C, ref, c)
        
        # if reftype == self.ctab['MOP']:
        #     return self.MOP_(reftype, C, ref, c) 

        # if reftype == self.ctab['DOP']:
        #     return self.DOP_(reftype, C, ref, c) 

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
        
        if C == self.ctab['XL']:       # N:XL -> 1 XL ⍝ Diamond
            return (rcat, (b, c))
                    
        raise NotImplementedError(f"Unknown category: {C}")
    
    def AF_(self, rcat: int, C: int, b: Any, c: Any) -> tuple:
        """
        Left node is curried dyadic function.
        """
        if C == self.ctab['A']:        # AF:A -> 6 A  ⍝ Monadic function application
            return (rcat, apply(_fun_ref(b), c))
        
        if C == self.ctab['F']:        # AF:F -> 5 F  ⍝ Derived function, atop
            return (rcat, atop(_fun_ref(b), _fun_ref(c)))
        
        if C == self.ctab['AF']:       # AF:AF -> 5 F  ⍝ Atop
            return (rcat, atop(_fun_ref(b), _fun_ref(c)))
        
        if C == self.ctab['MOP']:      # AF:MOP -> 8 F  ⍝ Mondadic operator with bound left function operand
            return (rcat, derive_function(_fun_ref(b), Voc.get_mop(c)))
        
        if C == self.ctab['XL']:       # AF:XL -> 1 XL ⍝ Diamond
            return (rcat, (b, c))
        
        raise NotImplementedError(f"Unknown category: {C}")

    def ASG_(self, rcat: int, C: int, b: Callable, c: Any) -> tuple:
        """
        Left node is a name assignment `a←`
        """
        if C in {
            self.ctab['A'],   self.ctab['F'],   self.ctab['N'], 
            self.ctab['H'],   self.ctab['AF'],  self.ctab['JOT'], 
            self.ctab['DOT'], self.ctab['MOP'], self.ctab['DOP']
        }:
            assert callable(b)
            return (rcat, b(c))
        
        raise NotImplementedError(f"Unknown category: {C}")
    
    def XL_(self, rcat: int, C: int, b: Callable, c: Any) -> tuple:
        """
        Left node is an expression list ...⋄...
        """
        if C in {self.ctab['A'], self.ctab['XL']}:
            return (rcat, (b, c))
        
        raise NotImplementedError(f"Unknown category: {C}")
    
    def eval(self, B: int, C: int, b: Any, c: Any) -> tuple:
        """
        Evaluate the expression b c
        """
        rcat = self.zmat[(B, C)]
        
        #---------- LEFT IS ARRAY -----------------
        if B == self.ctab['A']:
            return self.A_(rcat, C, b, c)

        #---------- LEFT IS FUNCTION --------------
        elif B == self.ctab['F']:
            return self.F_(rcat, C, b, c)
        
        #---------- LEFT IS NAME ------------------
        elif B == self.ctab['N']:
            return self.N_(rcat, C, b, c)
        
        #---------- LEFT IS CURRIED DYAD ----------
        elif B == self.ctab['AF']:
            return self.AF_(rcat, C, b, c)
        
        #---------- LEFT IS NAME ASSIGNMENT -------
        elif B == self.ctab['ASG']:
            return self.ASG_(rcat, C, b, c)
        
        #---------- LEFT IS EXPRESSION LIST ------
        elif B == self.ctab['XL']:
            return self.XL_(rcat, C, b, c)
                
        # return (rcat, (b, c))
        raise NotImplementedError(f"Unknown category: {B}")

def _name_is_array(name: Any) -> bool:
    return type(name) == str and name[0].islower()

def _name_is_function(name: Any) -> bool:
    return type(name) == str and name[0].isupper()

def _name_is_mop(name: Any) -> bool:
    return type(name) == str and len(name) > 1 and name[0] == '_' and name[1].isupper() and name[-1] != '_'

def _name_is_dop(name: Any) -> bool:
    return type(name) == str and len(name) > 1 and name[0] == '_' and name[1].isupper() and name[-1] == '_'

def _flat(l, acc):
    if l == (-1, -1):
        return acc
    acc.append(l[0])
    return _flat(l[1], acc)

def _simple_scalar(e: Any) -> bool:
    return isinstance(e, (int, float, complex, str))

def _payload(e: Any) -> APLTYPE:
    if type(e) == tuple:
        return e[1]
    return _ensure_array(e)

def _fun_ref(f: Any) -> Callable:
    if callable(f):
        return f
    return Voc.get_fn(f)

def _ensure_array(a: Any) -> np.ndarray:
    if type(a) == np.ndarray:
        return a
    return np.array(a)
            
def curried_assign(a: str) -> Callable:
    return lambda val: assign(a, val)

def assign(a: str, b: Callable|APLTYPE) -> Callable|APLTYPE:
    """
    a ← b
    """
    assert(type(a) == str)
    Env.set(a, b) # type: ignore 
    return b

def strand(left: tuple|APLTYPE, right: tuple|APLTYPE) -> np.ndarray:
    # Flat strand of two simple scalars: 1 2
    if _simple_scalar(left) and _simple_scalar(right): 
        return np.hstack((left, right)) 
    
    # Flat strand of more than two simple scalars: 1 2 3
    if type(left) == np.ndarray and _simple_scalar(right):
        return np.hstack((left, right))
    
    if _simple_scalar(left) and type(right) == np.ndarray:
        return np.hstack((left, right))
    
    if type(left) == tuple and type(right) == tuple:
        # Flat, bracketed, strand of two simple scalars: (1)(2)
        if _simple_scalar(left[1]) and _simple_scalar(right[1]):
            return np.hstack((left[1], right[1]))
        else: # Bracketed arrays, e,g (1 2 3)(4 5 6) or (1 2 3)4
            nested = np.empty(2, dtype=object)
            nested[:] = [_payload(left[1]), _payload(right[1])] # type: ignore
            return nested
        
    if type(left) == tuple:
        if type(right) == tuple:
            # Flat, bracketed, strand of two simple scalars: (1)(2)
            if _simple_scalar(left[1]) and _simple_scalar(right[1]):
                return np.hstack((left[1], right[1]))    
            # Bracketed arrays, e,g (1 2 3)(4 5 6)
            nested = np.empty(2, dtype=object)
            nested[:] = [_payload(left[1]), _payload(right[1])] # type: ignore
            return nested
        else:
            nested = np.empty(2, dtype=object)
            nested[:] = [_payload(left[1]), _payload(right)] # type: ignore
            return nested
    else:
        if type(right) == tuple:
            nested = np.empty(2, dtype=object)
            nested[:] = [_payload(left), _payload(right[1])] # type: ignore
            return nested
        else:
            nested = np.empty(2, dtype=object)
            nested[:] = [_payload(left), _payload(right)] # type: ignore
            return nested

def curry(left: np.ndarray, right: Callable) -> Callable:
    """
    Bind the left argument `left` to the dyadic function `right`, to give
    a monadic function.
    """
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
    v = left(None, right)
    print(f"Applying {left} to {right}: {v}")
    return v

def atop(left: Callable, right: Callable) -> Callable:
    """
    (f⍤g y) ≡ f g y
    x f⍤g y) ≡ (f x g y)
    """
    def derived(alpha: Optional[np.ndarray], omega: np.ndarray) -> np.ndarray:
        return left(None, right(alpha, omega))
    return derived

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
