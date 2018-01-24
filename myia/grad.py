
from .anf_ir import Apply, Constant, Parameter, Graph
from . import primops
from collections import defaultdict
from .anf_ir_utils import dfs
from .cconv import NestingAnalyzer


def is_constant(x):
    return isinstance(x, Constant)


def is_graph(x):
    return isinstance(x, Constant) and isinstance(x.value, Graph)


add = Constant(primops.Add())
J = Constant(primops.J())
index = Constant(primops.Index())
mktuple = Constant(primops.MakeTuple())
return_ = Constant(primops.Return())


class Grad:

    def __init__(self):
        self.todo = set()
        self.todo_bw = set()

        self.nest = NestingAnalyzer()
        self.fv_order = {}

        self.tagged_graphs = {}
        self.backpropagator_graphs = {}
        self.done_graphs = set()

        self.tagged_nodes = {}
        self.backpropagator_nodes = {}
        self.sensitivity_nodes = {}
        self.step_nodes = {}
        self.graph_to_ct = defaultdict(set)

    def scaffold_graph(self, graph):

        self.nest.run(graph)
        self.fv_order[graph] = list(self.nest.fvs[graph])

        gname = graph.debug.debug_name
        out = graph.return_.inputs[1]

        tgraph = Graph()
        tgraph.debug.name = f'↑{gname}'
        self.tagged_graphs[graph] = tgraph
        for p in graph.parameters:
            tp = Parameter(tgraph)
            tp.debug.name = f'↑{p.debug.debug_name}'
            tgraph.parameters.append(tp)
            self.tagged_nodes[p] = tp

        bgraph = Graph()
        bgraph.debug.name = f'♢{gname}'
        self.backpropagator_graphs[graph] = bgraph
        bparam = Parameter(bgraph)
        bparam.debug.name = f'∇{gname}'
        bgraph.parameters.append(bparam)
        self.sensitivity_nodes[(out, graph)] = bparam

        self.todo.add(graph)
        self.todo_bw.add(graph)

        return Constant(tgraph), Constant(bgraph)

    def process_graph(self, graph):
        self.process_all_graphs_forward(graph)
        self.process_all_graphs_backward()
        return self.tagged_graphs[graph]

    def process_graph_forward(self, graph):
        if graph in self.done_graphs:
            return

        if graph not in self.tagged_graphs:
            self.scaffold_graph(graph)

        tgraph = self.tagged_graphs[graph]
        bgraph = self.backpropagator_graphs[graph]

        tgraph.output = Apply([
            mktuple,
            self.phi(graph.return_.inputs[1]),
            Constant(bgraph)
        ], tgraph)

        self.done_graphs.add(graph)

    def process_all_graphs_forward(self, root):
        self.todo.add(root)
        while self.todo:
            g = self.todo.pop()
            self.process_graph_forward(g)

    def process_graph_backward(self, graph):
        bgraph = self.backpropagator_graphs[graph]
        bgraph.output = Apply([
            mktuple,
            Apply([
                mktuple,
                *[self.rho(p, graph)
                  for p in self.fv_order[graph]]
            ], bgraph),
            *[self.rho(p, graph)
              for p in graph.parameters]
        ], bgraph)

    def process_all_graphs_backward(self):
        for g in self.todo_bw:
            self.process_graph_backward(g)

    def phi(self, node):
        if node in self.tagged_nodes:
            return self.tagged_nodes[node]

        tg = node.graph and self.tagged_graphs[node.graph]

        if is_graph(node):
            tagged, bprop = self.scaffold_graph(node.value)
            self.graph_to_ct[node.value].add(node)
        elif is_constant(node):
            tagged, bprop = Apply([J, node], tg), None
        else:
            tagged_args = [self.phi(n) for n in node.inputs]
            app = Apply(tagged_args, tg)
            tagged = Apply([index, app, Constant(0)], tg)
            bprop = Apply([index, app, Constant(1)], tg)

        self.tagged_nodes[node] = tagged
        self.backpropagator_nodes[node] = bprop
        if node.debug.name:
            tagged.debug.name = f'↑{node.debug.name}'
            bprop.debug.name = f'♢{node.debug.name}'
        return tagged

    def bprop_step(self, node):
        if node in self.step_nodes:
            return self.step_nodes[node]

        bg = node.graph and self.backpropagator_graphs[node.graph]
        bprop = self.backpropagator_nodes[node]
        if bprop:
            rval = Apply([
                bprop,
                self.rho(node, node.graph)
            ], bg)
            self.step_nodes[node] = rval
            return rval
        else:
            return None

    def rho(self, node, graph):
        key = (node, graph)
        if key in self.sensitivity_nodes:
            return self.sensitivity_nodes[key]

        bg = self.backpropagator_graphs[graph]

        contribs = []

        for user, idx in node.uses:
            ug = user.graph
            if ug is graph:
                contrib = Apply([index,
                                self.bprop_step(user),
                                Constant(idx)], bg)
                contribs.append(contrib)
            elif self.nest.nested_in(ug, graph):
                for graph_ct in self.graph_to_ct[ug]:
                    fvs = self.fv_order[ug]
                    assert node in fvs
                    idx = fvs.index(node)
                    contrib = Apply([index,
                                    self.rho(graph_ct, graph),
                                    Constant(idx)], bg)
                    contribs.append(contrib)
            else:
                pass

        # NOTE: The order of nodes in contribs is not deterministic, because
        # the order of users isn't. In theory that doesn't matter, because we
        # add them all, but in practice there could be slight numerical
        # differences.

        if len(contribs) == 0:
            sens = Constant(0)  # TODO: should be zeros_like(node)
        elif len(contribs) == 1:
            sens, = contribs
        else:
            sens = Apply([add, *contribs], bg)
        self.sensitivity_nodes[node] = sens
        if node.debug.name:
            sens.debug.name = f'∇{node.debug.name}'
        return sens
