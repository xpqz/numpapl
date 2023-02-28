"""
Read and parse the grammar format as described in

    https://dfns.dyalog.com/s_Binding.htm
    https://dfns.dyalog.com/n_parse.htm
    https://dfns.dyalog.com/s_parse.htm

"""
from functools import reduce
from itertools import product
import re
from string import ascii_letters, digits, whitespace

import numpy as np

class Grammar:
    functions = '+-×÷*=≥>≠∨∧⍒⍋⌽⍉⊖⍟⍱⍲!?∊⍴~↑↓⍳○*⌈⌊∇⍎⍕⊃⊂∩∪⊥⊤|≡≢,⍪⊆⌹'
    hybrids = '/⌿\⍀'
    monadic_operators = '⌸¨⍨'
    dyadic_operators = '⍣⌺@⍥⍤'

    def __init__(self, cats, zmat, bmat, xmat, bkts, bcats, lbs, rbs, blabs):
        self.cats = cats
        self.zmat = zmat
        self.bmat = bmat
        self.xmat = xmat
        self.bkts = bkts
        self.bcats = bcats
        self.lbs = lbs
        self.rbs = rbs
        self.blabs = blabs

    @classmethod
    def compile(cls, grammar: list[str]):
        sep = [r.startswith('⍝⍝⍝⍝⍝⍝⍝⍝') for r in grammar]
        idx = [i for i, s in enumerate(sep) if s]

        categories = grammar[idx[0]:idx[1]]
        macros = grammar[idx[1]+1:idx[2]]
        bindings = grammar[idx[2]+1:]

        # Categories & brackets
        if categories[-1] == '': # Sometimes we have an empty line at the end :/
            categories = categories[:-1]

        cats = [r[:r.index(' ')] for r in categories[1:] if r != ''] # Last are the brackets
        brackets = None
        if cats[-1].startswith('()'):
            b = categories.pop(-1)
            brackets = b[:b.index('⍝')].strip()
            cats.pop(-1)

        lcats = len(cats)

        # Macros
        if macros[-1] == '': # Sometimes we have an empty line at the end :/
            macros = macros[:-1]
        macro_tab = dict([mk[:mk.index('⍝')].strip().split('=') for mk in macros]) # Trim

        # Bindings -- strip comments. Separating lines are sometimes empty, and sometimes
        # they contain a lamp :/
        bb = []
        for b in bindings:
            b = b.strip()
            if b == '':
                bb.append(b)
            else:
                bb.append(b[:b.index('⍝')].strip())

        bindings = bb

        # The bindings section has sub-sections, now delineated by a 
        # an empty line

        sep = [r=='' for r in bindings]
        idx = [i for i, s in enumerate(sep) if s]

        # Partition
        pidx = [e.tolist() for e in np.split(np.arange(len(bindings)), idx)]
        sections = []
        for sect in pidx:
            if len(sect) == 1:
                sections.append([bindings[sect[0]]])
            else:
                sections.append(bindings[sect[1]:sect[-1]+1])

        # Expand macros and separate multiple entries on space
        split_sections = []
        for s in sections:
            new_s = []
            for item in s:
                new_s.extend(macro_expand(item, macro_tab).split())
            split_sections.append(new_s)
                    
        # Expand the dots
        dot_sections = []
        for s in split_sections:
            new_s = []
            for item in s:
                new_s.extend(dot_expand(item))
            dot_sections.append(new_s)

        # Build the complete bindings table
        zmat = np.zeros(lcats*lcats, dtype=int).reshape((lcats, lcats)) # Resulting categories
        bmat = np.zeros(lcats*lcats, dtype=int).reshape((lcats, lcats)) # Binding strengths

        for i, s in enumerate(dot_sections):
            for (left, right), target in s:  # type: ignore
                l_index = cats.index(left)   # type: ignore
                r_index = cats.index(right)  # type: ignore
                t_index = cats.index(target) # type: ignore

                zmat[l_index, r_index] = t_index
                bmat[l_index, r_index] = len(dot_sections) - i

        assert brackets
        bkts = [a for a in brackets if a not in ascii_letters + digits + whitespace] # Bracket pairings
        lbs, rbs = np.array(list(zip(bkts[::2], bkts[1::2]))).T.tolist()             # Lefts and rights of brackets
        p = f'[{re.escape("".join(lbs))}](.*?)[{re.escape("".join(rbs))}]'             
        blabs = re.findall(p, brackets)                                              # Bracket labels
        bcats = []                                                                   # Bracket categories

        for b in blabs:
            try:
                bcats.append(cats.index(b))
            except ValueError:
                bcats.append(len(cats))

        xmat = np.pad(bmat, ((1, 1), (1, 1)), 'constant') # Extended bmat: pad with zeros in x and y

        return cls(cats, zmat, bmat, xmat, bkts, bcats, lbs, rbs, blabs)
    
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
            elif ch in Grammar.functions:
                pairs.append((self.cats.index('F'), ch))     # Primitive function
            elif ch in Grammar.hybrids:
                pairs.append((self.cats.index('H'), ch))     # Hybrid function/operator
            elif ch in Grammar.monadic_operators:          
                pairs.append((self.cats.index('MOP'), ch))   # Primitive monadic operator
            elif ch in Grammar.dyadic_operators:
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

        BbCc = (self.zmat[(B, C)], (b, c)) # BbCc←zmat[B;C],⊂b c ⍝ B bound with C.
        return self.bind(((D_, L), Aa, BbCc, R, _D)) # binds with token to the right?

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
          
def macro_expand(str, macros):
    """
    Expand macros in a string.
    """
    s = str
    while True:
        changed = False
        for k, v in macros.items():
            s_r = s.replace(k, v)
            if s_r != s:
                changed = True
                s = s_r
        if not changed:
            break
    return s

def _unpack(item: int|tuple) -> tuple:
    """
    Python doesn't do scalar extension, so we have to do it ourselves.
    """
    if type(item) == int:
        return (item, item)
    else:
        return item # type: ignore
    
def dot_expand(str):
    """
    A:F.H => AF

    means 

    A:F => AF
    A:H => AF
    """
    rule, target = str.split('→')
    left, right = rule.split(':')

    lefts = left.split('.')
    rights = right.split('.')
    targets = target.split('.')

    rules = list(product(lefts, rights))
    if len(targets) == 1: # Scalar extension on the right
        targets = targets * len(rules)
    if len(rules) != len(targets):
        raise SyntaxError('SYNTAX ERROR')
    return list(zip(rules, targets))

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
    with open('grammar') as f:
        grammar = f.read().splitlines()
    g = Grammar.compile(grammar)
    # Ok, we have parsed the grammar, yay: cats, zmat, bmat, bkts, blabs
    
    src = '{⍺+⍵}/⍳5'
    # src = '+/⍳5'
    # src = '⍳5'
    # src = 'x ← ⍳5 ⋄ x×3'
    pairs = g.classify(src)
    eos = -1
    ll = [eos]+pairs[:-2]
    if ll == [-1]:
        D_ = -1
    else:
        # Left cons list: ∆_←↑{⍺ ⍵}⍨/⌽eos,¯2↓pairs
        D_ = tuple(reduce(lambda x, y: (x, y), ll)) # type: ignore 
    Aa, Bb, Cc = (eos, eos, *pairs, eos)[-3:] # 3-token window
    stream = (D_, Aa, Bb, Cc, eos)
    tree = g.bind(stream)

    print(tree)
