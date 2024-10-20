"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable
# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


def mul(x: float, y: float) -> float:
    """Multiply `x` and `y`."""
    return x * y


def id(x: float) -> float:
    """Return `x`."""
    return x


def add(x: float, y: float) -> float:
    """Add `x` and `y`."""
    return x + y


def neg(x: float) -> float:
    """Return the negative of `x`."""
    return -1.0 * x


def lt(x: float, y: float) -> float:
    """Check if `x` is less than `y`."""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Check if `x` is equal to `y`."""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Return the maximum of `x` and `y`."""
    if x > y:
        return x
    else:
        return y


def is_close(x: float, y: float) -> float:
    """Check if `x` is within 1e-2 of `y`."""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Return the sigmoid of `x`."""
    if x >= 0:
        return 1.0 / (1.0 + exp(-x))
    else:
        return exp(x) / (1.0 + exp(x))


def relu(x: float) -> float:
    """Return the ReLU of `x`."""
    return x if x > 0 else 0.0


Epsilon = 1e-6


def log(x: float) -> float:
    """Return the natural logarithm of `x`."""
    return math.log(x + Epsilon)


def exp(x: float) -> float:
    """Return `e` raised to the power of `x`."""
    return math.exp(x)


def inv(x: float) -> float:
    """Return the inverse of `x`."""
    return 1.0 / x


def log_back(x: float, y: float) -> float:
    """Computes the derivative of log(`x`) times `y`, i.e. `y`/`x`."""
    return y / (x + Epsilon)


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of 1/`x` times `y`, i.e. -`y`/(`x`**2)."""
    return -(1.0 / (x**2)) * y


def relu_back(x: float, y: float) -> float:
    """Computes the derivative of the ReLU function at `x` times `y`, i.e. `y` if `x` > 0 else 0."""
    return y if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order function that applies a function `fn` to each element of an iterable.

    Args:
    ----
        fn: A function that takes a float and returns a float.

    Returns:
    -------
        A function that takes an iterable of floats and returns an iterable of floats.

    """

    def _map(xs: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in xs:
            ret.append(fn(x))
        return ret

    return _map


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order function that combines two iterables elementwise using a function `fn`.

    Args:
    ----
        fn: A function that takes two floats and returns a float.

    Returns:
    -------
        A function that takes two iterables of floats and returns an iterable of floats.

    """

    def _zipWith(xs: Iterable[float], ys: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(xs, ys):
            ret.append(fn(x, y))
        return ret

    return _zipWith


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Higher-order function that reduces an iterable to a single value using a function `fn`.

    Args:
    ----
        fn: A function that takes two floats and returns a float.
        start: The initial value for the reduction.

    Returns:
    -------
        A function that takes an iterable of floats and returns a float.

    """

    def _reduce(xs: Iterable[float]) -> float:
        ret = start
        for x in xs:
            ret = fn(ret, x)
        return ret

    return _reduce


def negList(xs: Iterable[float]) -> Iterable[float]:
    """Negate a list using `map` and `neg`."""
    return map(neg)(xs)


def addLists(xs: Iterable[float], ys: Iterable[float]) -> Iterable[float]:
    """Add two lists elementwise using `zipWith` and `add`."""
    return zipWith(add)(xs, ys)


def sum(xs: Iterable[float]) -> float:
    """Sum a list using `reduce` and `add`."""
    return reduce(add, 0.0)(xs)


def prod(xs: Iterable[float]) -> float:
    """Produce a list using `reduce` and `mul`."""
    return reduce(mul, 1.0)(xs)
