import math
import numpy as np
import pytest 

from apl import APL
import arr

def nested_array_equal(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape != b.shape:
            return False
        return np.all([nested_array_equal(ai, bi) for ai, bi in zip(a.flat, b.flat)])
    else:
        return a == b

def nest(shape: tuple[int, int], arr: list[list]) -> np.ndarray:
    bound = math.prod(shape)
    a = np.empty(bound, dtype=object)
    a[:] = [np.array(e) for e in arr]
    return a.reshape(shape)

def run(src):
    return(APL().run(src))

scalar_arithmetic = [
    ("+2", 2),
    ("¯2", -2),
    ("+3j4", 3-4j),
    ("1 + 2", 3),
    ("1 ÷ 2", 1/2),
    ("1 - 2 - 3 - 4", -2),
    ("×¯76", -1),
    ("×0", 0),
    ("×76", 1),
    ("÷5 ⍝ Reciprocal", 1/5),
]

@pytest.mark.parametrize("test_input,expected", scalar_arithmetic)
def test_arithmetic(test_input, expected):
    assert run(test_input) == expected

vector_arithmetic = [
    ("1+1 2 3", np.array([2, 3, 4])),
    ("1 2 3+1", np.array([2, 3, 4])),
    ("1 2 3+1 2 3", np.array([2, 4, 6])),
    ("2 4 6-1 2 3+1 2 3", np.array([0, 0, 0])),
    ("1 2 3×1 2 3", np.array([1, 4, 9])),
    ("÷1 2 3", np.array([1, 0.5, 1/3])),
]

@pytest.mark.parametrize("test_input,expected", vector_arithmetic)
def test_vector_arithmetic(test_input, expected):
    assert np.array_equal(run(test_input), expected)

array_structural = [
    ("2 2⍴1 2 3 4", np.array([1, 2, 3, 4]).reshape((2, 2))),
    ("2 2⍴1", np.array([1, 1, 1, 1]).reshape((2, 2))),
    ("2 2⍴1 2 3 4 5 6 7", np.array([1, 2, 3, 4]).reshape((2, 2))),
    ("⍉2 2⍴1 2 3 4", np.array([1, 3, 2, 4]).reshape((2, 2))),
    ("⍳5", np.arange(5)),
    ("⍳2 2", nest((2, 2), [[0, 0], [0, 1], [1, 0], [1, 1]])),
]

@pytest.mark.parametrize("test_input,expected", array_structural)
def test_array_structural(test_input, expected):
    assert nested_array_equal(run(test_input), expected)

hybrids = [
    ("+/⍳10", 45),
    ("+⌿⍳10", 45),
    ("-/⍳10", -5),
]

@pytest.mark.parametrize("test_input,expected", hybrids)
def test_hybrids(test_input, expected):
    assert nested_array_equal(run(test_input), expected)

primitives = [
    # (",4", np.array([4])),
    # ("2⊢4", 4),
    # ("2⊣4", 2),
    # ("1 2 3~2", np.array([1, 3])),
    # ("1 2 3,4", np.array([1, 2, 3, 4])),
    # ("1 2 3,⊂4", np.array([1, 2, 3, 4])),
    # ("⊂,3", arr.enclose(np.array([3]))),
    # ("1 2 3,⊂,4", np.array([1, 2, 3, np.array([4])], dtype=object)),
    # ("⊃⊂,4", np.array([4])),
    # ("(2 3⍴1 2 3 4 5 6),1", np.array([1,2,3,1,4,5,6,1]).reshape((2, 4))),
    # ("2 ⊥ 1 1 0 1", 13),
    # ("24 60 60 ⊥ 2 46 40", 10_000),
    # ("(4 3⍴1 1 1 2 2 2 3 3 3 4 4 4)⊥3 8⍴0 0 0 0 1 1 1 1 0 0 1 1 0 0 1 1 0 1 0 1 0 1 0 1", np.array([0,1,1,2,1,2,2,3,0,1,2,3,4,5,6,7,0,1,3,4,9,10,12,13,0,1,4,5,16,17,20,21]).reshape((4, 8))),
    # ("24 60 60 ⊤ 10000", np.array([2, 46, 40])),
    # ("2 2 2 2 ⊤ 5 7 12", np.array([0,0,1,1,1,1,0,1,0,1,1,0]).reshape((4,3))),
    # ("⍋ 33 11 44 66 22", np.array([1,4,0,2,3])),
    # ("⍒ 33 11 44 66 22", np.array([3,2,0,4,1])),
    # ("1 0 2/1 2 3", np.array([1,3,3])),
    # ("1 0 2⌿1 2 3", np.array([1,3,3])),
    # ("5/1", np.array([1,1,1,1,1])),
    # ("5/1 2 3", np.array([1,1,1,1,1,2,2,2,2,2,3,3,3,3,3])),
    # ("≢2 2⍴1", 2),
    # ("≢2", 1),
    # ("≢2 2", 2),
    # ("1 1≢2 2", 1),
    # ("2 2≢2 2", 0),
    # ("~2 2≢2 2", 1),
    # ("2 2≢1 2⍴2 2", 1),
    # ("2≢1", 1),
    # ("1 1≡2 2", 0),
    # ("2 2≡2 2", 1),
    # ("2 2≡1 2⍴2 2", 0),
    # ("2≡1", 0),
    # ("'mississippi'~'s'", np.array(list('miiippi'))),
    # ("1 0 2⍉3 3 3⍴1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3", np.array([1,1,1,2,2,2,3,3,3,1,1,1,2,2,2,3,3,3,1,1,1,2,2,2,3,3,3]).reshape((3,3,3))),
    ("3↑1 2 3 4 5", np.array([1, 2, 3])),
    ("3 3↑9 9⍴⍳81", np.array([0,1,2,9,10,11,18,19,20]).reshape((3,3))),
    ("3↑1", np.array([1,0,0])),
    ("3 3↑1", np.array([1,0,0,0,0,0,0,0,0]).reshape((3,3))),
    ("3 ¯3↑9 9⍴⍳81", np.array([6,7,8,15,16,17,24,25,26]).reshape((3,3))),
    ("¯3↑1", np.array([0,0,1])),
    ("3 ¯3↑1", np.array([0,0,1,0,0,0,0,0,0]).reshape((3,3))),
]

@pytest.mark.parametrize("test_input,expected", primitives)
def test_primitives(test_input, expected):
    assert nested_array_equal(run(test_input), expected)
