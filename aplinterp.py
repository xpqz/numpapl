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

