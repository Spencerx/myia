"""Utilities for manipulating and inspecting the IR."""
from typing import Iterable

from myia.anf_ir import Graph, ANFNode, Constant, Parameter, Apply


def dfs(root: ANFNode, follow_graph: bool = False) -> Iterable[ANFNode]:
    """Perform a depth-first search."""
    seen = set()
    to_visit = [root]
    while to_visit:
        node = to_visit.pop()
        seen.add(node)
        yield node
        for in_ in node.incoming:
            if in_ not in seen:
                to_visit.append(in_)
        if isinstance(node.value, Graph) and follow_graph:
            if node.value.return_ not in seen:
                to_visit.append(node.value.return_)


def is_apply(x):
    return isinstance(x, Apply)


def is_parameter(x):
    return isinstance(x, Parameter)


def is_constant(x):
    return isinstance(x, Constant)


def is_graph_constant(x):
    return isinstance(x, Constant) and isinstance(x.value, Graph)
