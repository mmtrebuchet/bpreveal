#!/usr/bin/env python3
"""Implements a small interpreter.

Syntax:

This interpreter interprets a subset of the Python programming language,
with an extension to the behavior of default arguments in lambdas. Since it
uses the Python parser, it obeys Python's operator precedence. If it
encounters a name in the expression, it looks it up in the supplied
environment.

Here are the supported operations:

  * Strings (with support for the .format method)
  * Integers and floats.
  * ``True``, ``False``, and ``None``.
  * Lists and list comprehensions
  * Dictionaries and dict comprehensions
  * Indexing for lists and dictionaries.
  * Unary ``not``, negation.
  * Binary multiplication, division, exponentiation, addition, subtraction.
  * Boolean ``and`` and ``or``.
  * Comparison with ``<``, ``>``, ``<=``, ``>=``, ``==``, and ``!=``.
  * Chained comparison, like ``0.2 < x < 0.5``.
  * Builtin functions: ``exp``, ``log``, ``sqrt``, ``abs``, ``any``,
    ``all``, ``min``, ``max``, ``sum``, ``len``, ``range``.
  * Builtin constants: ``e``, ``pi``, ``true``, ``false``, and ``null``.
    The ``true``, ``false``, and ``null`` constants are for compatibility
    with JSON. The Python literals ``True``, ``False``, and ``None`` are also
    supported.
  * User-defined functions with ``lambda`` syntax. By abusing default parameters,
    you can even create recursive functions!

The extension to the lambda default parameters lets you use letrec-style
recursive functions without needing to use the Z combinator to implement
recursion. For example, this is valid in my interpreter but not valid
Python::

    (lambda x=5, y=(lambda q: x + q): y(7))()

In effect, you can think of this being translated to::

    (letrec ((x 5) (y (lambda (q) (+ x q)))) (y 7))

Note that if you use a lambda as a default parameter to a lambda,
you cannot override any default arguments when you call it. So this::

    (lambda x=5, y=(lambda q: x + q): y(7))(3)

is an error because the call to the outer lambda attempted to specify ``x``.

.. highlight:: none

.. literalinclude:: ../../doc/bnf/interpreter.bnf

"""
import ast
import math
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any, TypeAlias, Union
# pylint: disable=consider-alternative-union-syntax
EVAL_RET_T: TypeAlias = Union[int, float, bool, str, Callable, "Closure",
                              dict, list, Collection, None]
# pylint: enable=consider-alternative-union-syntax
ENV_T: TypeAlias = dict[str, EVAL_RET_T]


class Closure:
    """Represents a function and its environment.

    You do not need to use this, it's only necessary to set up letrec-style
    default parameters for lambdas.
    """

    def __init__(self, env: ENV_T, args: ast.arguments, body: ast.expr):
        self.env = env.copy()
        self.body = body
        self.args = args
        self.defaults = {}
        self.usingLetrec = False
        defaultStartIdx = len(args.args) - len(args.defaults)
        for i in range(defaultStartIdx, len(args.args)):
            self.defaults[i] = evalAstRaw(args.defaults[i - defaultStartIdx], env)
        # Implement letrec-style self-reference in default args.
        for argVal in self.defaults.values():
            if isinstance(argVal, Closure):
                self.usingLetrec = True
                for aIdx, aVal in self.defaults.items():
                    aName = self.args.args[aIdx].arg
                    argVal.env[aName] = aVal

    def run(self, argList: list[EVAL_RET_T]) -> EVAL_RET_T:
        """Actually evaluate the body of the lambda."""
        if len(argList) + len(self.defaults) > len(self.args.args):
            # We have defaults that are being overridden.
            if self.usingLetrec:
                raise SyntaxError("Cannot override default params when one is a lambda.")

        innerEnv = self.env.copy()
        for i, arg in enumerate(self.args.args):
            argName = arg.arg
            if len(argList) > i:
                innerEnv[argName] = argList[i]
            else:
                innerEnv[argName] = self.defaults[i]
        return evalAstRaw(self.body, innerEnv)


def _evalCompare(left: ast.expr, ops: list[ast.cmpop],  # pylint: disable=too-many-return-statements
                 comparators: list[ast.expr], env: ENV_T) -> EVAL_RET_T:
    prev = evalAstRaw(left, env)
    for op, rhs in zip(ops, comparators):
        rv = evalAstRaw(rhs, env)
        # This looks weird. Why don't I just
        # case ast.Lt():  # flake8: noqa: E800
        #    return prev < rv  # flake8: noqa: E800
        # ?
        # Because there's a loop, and we need to go
        # over every comparison if the user supplied
        # a < b < c.
        match op:
            case ast.Lt():
                if prev >= rv:  # type: ignore
                    return False
            case ast.Gt():
                if prev <= rv:  # type: ignore
                    return False
            case ast.LtE():
                if prev > rv:  # type: ignore
                    return False
            case ast.GtE():
                if prev < rv:  # type: ignore
                    return False
            case ast.NotEq():
                if prev == rv:
                    return False
            case ast.Eq():
                if prev != rv:
                    return False
            case ast.In():
                if prev not in rv:  # type: ignore
                    return False
            case ast.NotIn():
                if prev in rv:  # type: ignore
                    return False
            case _:
                raise SyntaxError(f"Unsupported comparison operator in expression: {op}")
        # Go to next comparison.
        prev = rv
    return True


def _evalBinary(left: ast.expr, op: ast.operator,  # pylint: disable=too-many-return-statements
                right: ast.expr, env: ENV_T) -> EVAL_RET_T:
    lhs = evalAstRaw(left, env)
    rhs = evalAstRaw(right, env)
    assert isinstance(lhs, (int, float, bool, str))
    assert isinstance(rhs, (int, float, bool, str))
    match op:
        case ast.Add():
            return lhs + rhs  # type: ignore
        case ast.Sub():
            return lhs - rhs  # type: ignore
        case ast.Mult():
            return lhs * rhs  # type: ignore
        case ast.Div():
            return lhs / rhs  # type: ignore
        case ast.Mod():
            return lhs % rhs  # type: ignore
        case ast.Pow():
            return lhs ** rhs  # type: ignore
        case ast.FloorDiv():
            return lhs // rhs  # type: ignore
        case _:
            raise SyntaxError(f"Unsupported binary operator in expression: {op}")


def _evalBool(op: ast.boolop, values: list[ast.expr], env: ENV_T) -> EVAL_RET_T:
    match op:
        case ast.And():
            for v in values:
                if not evalAstRaw(v, env):
                    return False
            return True
        case ast.Or():
            for v in values:
                if evalAstRaw(v, env):
                    return True
            return False
        case _:
            raise SyntaxError(f"Unsupported boolean operator in expression: {op}")


def _evalUnary(op: ast.unaryop, operand: ast.expr, env: ENV_T) -> EVAL_RET_T:
    match op:
        case ast.USub():
            ret = evalAstRaw(operand, env)
            assert isinstance(ret, (int, float, bool))
            return -ret
        case ast.Not():
            return not evalAstRaw(operand, env)
        case _:
            raise SyntaxError(f"Unsupported unary operator in expression: {op}")


def _evalLambda(args: ast.arguments, body: ast.expr, env: ENV_T) -> Closure:
    c = Closure(env, args, body)
    return c


def _evalCall(func: ast.expr, args: list[ast.expr], env: ENV_T) -> EVAL_RET_T:
    fun = evalAstRaw(func, env)
    argVals = [evalAstRaw(arg, env) for arg in args]
    if isinstance(fun, Callable):
        return fun(argVals)
    if isinstance(fun, Closure):
        return fun.run(argVals)
    raise ValueError(f"Function {fun}  is not a closure or callable.")


def _evalIf(test: ast.expr, body: ast.expr, orelse: ast.expr, env: ENV_T) -> EVAL_RET_T:
    return evalAstRaw(body, env) if evalAstRaw(test, env) else evalAstRaw(orelse, env)


def _evalSubscript(value: ast.expr, sliceExpr: ast.expr, env: ENV_T) -> EVAL_RET_T:
    theCollection = evalAstRaw(value, env)
    theSlice = evalAstRaw(sliceExpr, env)
    assert isinstance(theCollection, Mapping | Sequence)
    assert not isinstance(theSlice, Closure)
    return theCollection[theSlice]  # type: ignore


def _evalDict(keys: list[ast.expr], values: list[ast.expr],
              env: ENV_T) -> dict[EVAL_RET_T, EVAL_RET_T]:
    return {evalAstRaw(keys[i], env): evalAstRaw(values[i], env) for i in range(len(keys))}


def _evalList(elts: list[ast.expr], env: ENV_T) -> list[EVAL_RET_T]:
    return [evalAstRaw(x, env) for x in elts]


def _evalListComp(elt: ast.expr, generators: list[ast.comprehension],
                  env: ENV_T) -> list[EVAL_RET_T]:
    if len(generators) == 0:
        return [evalAstRaw(elt, env)]
    env = env.copy()
    ret = []
    gen = generators[0]
    assert isinstance(gen.target, ast.Name)
    target = gen.target.id
    iterable = evalAstRaw(gen.iter, env)
    assert isinstance(iterable, Collection)
    for val in iterable:
        env[target] = val
        doInsert = True
        for condition in gen.ifs:
            if not evalAstRaw(condition, env):
                doInsert = False
        if not doInsert:
            continue
        ret.extend(_evalListComp(elt, generators[1:], env))
    return ret


def _evalDictComp(key: ast.expr, value: ast.expr, generators: list[ast.comprehension],
                  env: ENV_T) -> dict[EVAL_RET_T, EVAL_RET_T]:
    if len(generators) == 0:
        return {evalAstRaw(key, env): evalAstRaw(value, env)}  # type: ignore
    env = env.copy()
    ret = {}
    gen = generators[0]
    assert isinstance(gen.target, ast.Name)
    target = gen.target.id
    iterable = evalAstRaw(gen.iter, env)
    assert isinstance(iterable, Collection)
    for val in iterable:
        env[target] = val
        doInsert = True
        for condition in gen.ifs:
            if not evalAstRaw(condition, env):
                doInsert = False
        if not doInsert:
            continue
        subComp = _evalDictComp(key, value, generators[1:], env)
        for k, v in subComp.items():
            ret[k] = v
    return ret


def _evalAttr(value: ast.expr, attr: str, env: ENV_T) -> EVAL_RET_T:
    evalValue = evalAstRaw(value, env)
    match attr:
        case "format":
            assert isinstance(evalValue, str)
            return lambda args: evalValue.format(*args)
        case _:
            raise SyntaxError(f"Attribute {attr} is not supported.")


def _readEnv(idStr: str, env: ENV_T) -> EVAL_RET_T:
    return env[idStr]


def _readConstant(value: EVAL_RET_T, env: ENV_T) -> EVAL_RET_T:
    del env  # env is unused for a constant.
    return value


def evalAstRaw(t: ast.AST,  # pylint: disable=too-many-return-statements
               env: dict[str, Any]) -> EVAL_RET_T:
    """Evaluates the (ast.parse()d) filter string t using variables in the environment env.

    :param t: The parsed AST that should be evaluated
    :param env: The environment containing the variables used in the expression.
    :return: The value of the expression.

    You should probably call :py:func:`~evalAst` instead of this, since that function
    will add helpful builtins to your environment.

    """
    match t:
        case ast.Dict(keys, values):
            return _evalDict(keys, values, env)  # type: ignore
        case ast.Constant(value):
            return _readConstant(value, env)
        case ast.Name(idStr):
            return _readEnv(idStr, env)
        case ast.Expr(value):
            return evalAstRaw(value, env)
        case ast.Compare(left, ops, comparators):
            return _evalCompare(left, ops, comparators, env)
        case ast.BinOp(left, op, right):
            return _evalBinary(left, op, right, env)
        case ast.Lambda(args, body):
            return _evalLambda(args, body, env)
        case ast.Call(func, args, _):
            return _evalCall(func, args, env)
        case ast.Attribute(value, attr, _):
            return _evalAttr(value, attr, env)
        case ast.BoolOp(op, values):
            return _evalBool(op, values, env)
        case ast.UnaryOp(op, operand):
            return _evalUnary(op, operand, env)
        case ast.Subscript(value, sliceExpr, _):
            return _evalSubscript(value, sliceExpr, env)
        case ast.List(elts, _):
            return _evalList(elts, env)
        case ast.ListComp(elt, generators):
            return _evalListComp(elt, generators, env)
        case ast.DictComp(key, value, generators):
            return _evalDictComp(key, value, generators, env)
        case ast.IfExp(test, body, orelse):
            return _evalIf(test, body, orelse, env)
        case ast.Module(body, _):
            if len(body) != 1:
                raise SyntaxError("ast contains more than one expression.")
            return evalAstRaw(body[0], env)
        case _:
            raise SyntaxError(f"Unable to interpret {ast.dump(t, indent=4)}, ({ast.unparse(t)})")


_BUILTIN_DICT = {}


def _buildFunctionDictionary() -> None:
    global _BUILTIN_DICT  # pylint: disable=global-variable-not-assigned

    def toStar(f: Callable) -> Callable:
        return lambda x: f(*x)
    _BUILTIN_DICT["exp"] = toStar(math.exp)
    _BUILTIN_DICT["log"] = toStar(math.log)
    _BUILTIN_DICT["sqrt"] = toStar(math.sqrt)
    _BUILTIN_DICT["pi"] = math.pi
    _BUILTIN_DICT["e"] = math.e
    _BUILTIN_DICT["abs"] = toStar(abs)
    _BUILTIN_DICT["any"] = toStar(any)
    _BUILTIN_DICT["all"] = toStar(all)
    _BUILTIN_DICT["min"] = toStar(min)
    _BUILTIN_DICT["sum"] = toStar(sum)
    _BUILTIN_DICT["max"] = toStar(max)
    _BUILTIN_DICT["len"] = toStar(len)
    _BUILTIN_DICT["range"] = toStar(range)
    _BUILTIN_DICT["true"] = True
    _BUILTIN_DICT["false"] = False
    _BUILTIN_DICT["null"] = None


_buildFunctionDictionary()


def _addFunctionsToEnv(env: ENV_T) -> ENV_T:
    return _BUILTIN_DICT | env


def evalAst(t: ast.AST, env: ENV_T | None = None, addFunctions: bool = True) -> EVAL_RET_T:
    """Evaluates the (ast.parse()d) filter string t using variables in the environment env.

    :param t: The parsed AST that should be evaluated
    :param env: The environment containing the variables used in the expression.
    :param addFunctions: If True (the default), then the functions
        ``exp``, ``log``, ``sqrt``, ``abs``, and ``len``, along with the
        constants ``pi`` and ``e`` will be added to the environment (but they
        will be shadowed by existing declarations if they have already been
        defined.)
    :return: The value of the expression.
    """
    if env is None:
        env = {}
    if addFunctions:
        env = _addFunctionsToEnv(env)
    return evalAstRaw(t, env)


def evalString(expr: str, env: ENV_T | None = None, addFunctions: bool = True) -> EVAL_RET_T:
    """Parses the given string and then runs evalAst on it."""
    if env is None:
        env = {}
    t = ast.parse(expr)
    return evalAst(t, env, addFunctions)


def evalFile(fname: str, env: ENV_T | None = None, addFunctions: bool = True) -> EVAL_RET_T:
    """Read in the named file and evaluate it."""
    if env is None:
        env = {}
    with open(fname, "r") as fp:
        contents = fp.read()
        return evalString(contents, env, addFunctions)


def _testAst() -> None:

    exprs = [["2 + 2", {}, 2, None],
             ["x - 1", {"x": 3}, 2, None],
             ["1 if 3 > 2 else 0", {}, 1, None],
             ["(lambda x: x + 1)(3)", {}, 4, None],
             ["(lambda x, y: y(x - 1))(3, lambda a: a*3)", {}, 6, None],
             ["(lambda x, y: y(x, y)) (5, lambda a, b: 1 if a <= 1 else a * b(a-1, b))",
              {}, 120, None],
             ["((lambda x: lambda x, y: y(x, y))(0)) "
              "(5, lambda a, b: 1 if a <= 1 else a * b(a-1, b))", {}, 120, None],
             ["sqrt(4)", {}, 2, None],
             ["len(s * 3)", {"s": "aoeu"}, 12, None],
             ["(lambda x, y=3: x + y)(2)", {}, 5, None],
             ["(lambda x, y=3: x + y)(2, 9)", {}, 11, None],
             ["(lambda x=5, y=(lambda q: x): x + y(0))()", {}, 10, None],
             ["(lambda x=5, y=(lambda n: 1 if n <= 1 else n * y(n - 1)): y(x))()", {}, 120, None],
             ["(lambda x, y=(lambda n: 1 if n <= 1 else n * y(n - 1)): y(x))(5)", {}, 120, None],
             ["(lambda n, isOdd=(lambda n: True if n == 1 else not isEven(n - 1)), "
              "isEven=(lambda n: True if n == 0 else not isOdd(n - 1)): isOdd(n))(5)",
              {}, True, None],
             ["(lambda n, isOdd=(lambda n: False if n == 0 else not isEven(n - 1)), "
              "isEven=(lambda n: True if n == 0 else not isOdd(n - 1)): isOdd(n))(4)",
              {}, False, None],
             ["(lambda n, isOdd=(lambda n: False if n == 0 else not isEven(n - 1)), "
              "isEven=(lambda n: True if n == 0 else not isOdd(n - 1)): isOdd(n))(100)",
              {}, False, None],
             ["(lambda a=5, b=(lambda: a): b)()()", {}, 5, None],
             ["(lambda a=5, b=(lambda: a): b)(6)()", {}, 5, SyntaxError],
             ["[1, 2, 3]", {}, [1, 2, 3], None],
             ["range(5)", {}, range(5), None],
             ["[x for x in range(5)]", {}, [0, 1, 2, 3, 4], None],
             ["[x + y for x in range(2) for y in range(3)]", {}, [0, 1, 2, 1, 2, 3], None],
             ["[y for x in [[1, 2], [3, 4]] for y in x]", {}, [1, 2, 3, 4], None],
             ["{y: x for x in [[1, 2], [3, 4]] for y in x if y > 2}",
              {}, {3: [3, 4], 4: [3, 4]}, None],
             ["[1, 2, 3][1]", {}, 2, None],
             ["(lambda x: x)\n(1)", {}, None, SyntaxError],
             ['"{0:s}h{0:s}".format("aoeu")', {}, "aoeuhaoeu", None],
             ["(lambda x: [x[0.2], x['aoeu']])({0.2: 5, 'aoeu': 'htns'})", {}, [5, "htns"], None],
             ["{}", {}, {}, None],
             ]

    for s, e, res, exType in exprs:
        evalRet = ""
        r = ""
        print(f"Test case: {s=}, {e=}, {res=}")  # noqa: T201
        try:
            r = evalString(s, e)
            if exType is not None:
                raise OverflowError(
                    f"Expected exception {exType} but it was not raised and got {r}.")
        except BaseException as ex:  # pylint: disable=broad-exception-caught
            if exType is type(ex):
                print("        Successfully raised exception.")  # noqa: T201
            else:
                raise
        if exType is None:
            prefix = "    "
            try:
                evalRet = eval(s, e)  # pylint: disable=eval-used
            except BaseException as ex:  # pylint: disable=broad-exception-caught  # noqa: B036
                print(f"{prefix}    Eval exception: {ex}")  # noqa: T201
            if r == res:
                prefix = prefix + "  "
            else:
                prefix = prefix + "XX"
            if evalRet == res:
                prefix = prefix + "  "
            else:
                prefix = prefix + "OO"
            print(f"{prefix}Returned {r=}, {res=}, {evalRet=}")  # noqa: T201


if __name__ == "__main__":
    _testAst()
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
