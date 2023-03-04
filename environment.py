from string import ascii_letters
from typing import Any, Callable

import numpy as np

class Env:
    _env: list[dict[str, np.ndarray|Callable]] = [ {
        '⍬': np.array([]),
        '⎕IO': np.array(0),
        '⎕A': np.array(ascii_letters[26:]),
        '⎕D': np.arange(10),
    } ]

    @classmethod
    def system_arrs(cls) -> set[str]:
        return {'⎕IO', '⎕A', '⎕D'}

    @classmethod
    def append(cls, env: dict) -> None:
        """
        Add a new environment
        """
        cls._env.append(env)

    @classmethod
    def pop(cls) -> None:
        """
        Remove the last environment
        """
        if len(cls._env) > 1:
            cls._env.pop()

    @classmethod
    def get(cls, key: str) -> Any:
        """
        Return the value stored at the first occurrence of `key`, 
        working backwards across all enviroments. Auto-shadows ⍺ and ⍵.
        """
        if key in '⍺⍵': # ⍺ and ⍵ are always local only
            try:
                return cls._env[-1][key]
            except KeyError:
                raise ValueError(f'VALUE ERROR: Undefined name: {key}')

        for env in reversed(cls._env):
            if key in env:
                return env[key]
        raise ValueError(f'VALUE ERROR: Undefined name: {key}')

    @classmethod
    def set(cls, key: str, val: np.ndarray|Callable) -> None:
        """
        Set a key-value pair in the current environment
        """
        cls._env[-1][key] = val

    @classmethod
    def amend(cls, key: str, fun: Callable, omega: np.ndarray) -> np.ndarray:
        """
        Apply `fun(v, omega)` to the value v stored at the first occurrence of `key`, 
        working backwards across all enviroments,

            a ⊢← 23
            a +← 6
        """
        if key.upper() in {'⍬', '⎕IO', '⎕A', '⎕D'}:
            raise SyntaxError('SYNTAX ERROR')

        for env in reversed(cls._env):
            if key in env:
                v = fun(env[key], omega)
                env[key] = v
                return v
        raise ValueError(f'VALUE ERROR: Undefined name: {key}')

    @classmethod
    def resolve(cls, a):
        if type(a) == str:
            return cls.get(a)
        return a



