from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # TODO: Implement for Task 1.1.
    # convert vals to list to allow element modification
    vals_plus = list(vals)
    vals_minus = list(vals)
    # modify the arg-th element of vals_plus and vals_minus
    vals_plus[arg] += epsilon
    vals_minus[arg] -= epsilon
    # calculate the central difference
    return (f(*vals_plus) - f(*vals_minus)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate the derivative for this variable."""
        ...

    @property
    def unique_id(self) -> int:
        """Return the unique identifier for the variable."""
        ...

    def is_leaf(self) -> bool:
        """Return True if the variable is a leaf node."""
        ...

    def is_constant(self) -> bool:
        """Return True if the variable is a constant."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Return the parents of the variable."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Return the chain rule results for the variable."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # TODO: Implement for Task 1.4.
    visited = set()
    topo_order = []

    def dfs(v: Variable) -> None:
        if v.unique_id in visited:
            return
        visited.add(v.unique_id)
        if not v.is_constant():
            for parent in v.parents:
                dfs(parent)
            topo_order.append(v)

    dfs(variable)
    topo_order.reverse()  # use reverse instead of return Reverse(topo_order) to avoid exhaustion of iterator
    return topo_order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    Returns:
    -------
        No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    # TODO: Implement for Task 1.4.
    topo_order = topological_sort(variable)
    # print("start backpropagation from variable", variable.unique_id, variable)
    # Initialize a dictionary to store gradients of each variable
    derivative_map = {}  # use unique_id as key to avoid hashable error
    derivative_map[variable.unique_id] = deriv
    # print(f"derivative map: {derivative_map}, topological order: {topo_order}")
    # perform backpropagation in topological order
    for var in topo_order:
        print(
            f"Backpropagating through variable {var.unique_id} with gradient {derivative_map[var.unique_id]}"
        )
        if var.is_leaf():
            # print(f"Accumulating derivative for leaf node {var.unique_id}: {d_output}")
            d_output = derivative_map[var.unique_id]
            var.accumulate_derivative(d_output)  # write the derivative to the leaf node
        else:
            # calculate parent's gradient using chain rule and update the derivative map
            for parent, local_grad in var.chain_rule(derivative_map[var.unique_id]):
                # print(f"Updating parent {parent.unique_id} with local gradient {local_grad}")
                if parent.unique_id in derivative_map:
                    derivative_map[parent.unique_id] += local_grad
                else:
                    derivative_map[parent.unique_id] = local_grad


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Return the saved values."""
        return self.saved_values
