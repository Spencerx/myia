
from myia.utils import Registry
from myia.api import parse
from myia.info import NamedDebugInfo, About
from myia.anf_ir import Graph, Apply, Constant, Parameter
from myia import primops
from myia.anf_ir_utils import replace
from myia.py_implementations import \
    Jinv, J, zeros_like, cons_tuple, head, tail, getitem, setitem


def transform_bprop(prim, fn):
    info = NamedDebugInfo(prim=prim, name=prim.name)

    bprop = parse(fn)
    bprop.debug.name = None
    bprop.debug.about = About(info, 'grad_bprop')
    # empty_tuple = Apply([Constant(primops.make_tuple)], bprop)
    # bprop.output.inputs.insert(1, empty_tuple)
    bprop.output = Apply([Constant(primops.cons_tuple),
                          Constant(()),
                          bprop.output], bprop)

    *args, dout = bprop.parameters

    with About(info, 'grad_fw'):
        outer = Graph()
        outer.primal = prim

    def app(prim, *args):
        return Apply([Constant(prim), *args], outer)

    transf_args = []
    for p in args:
        with About(p.debug, 'grad_fw'):
            outer_p = Parameter(outer)
            outer.parameters.append(outer_p)
        replace(p, outer_p)
        transf_args.append(app(primops.Jinv, outer_p))

    with About(dout.debug, 'grad_bw'):
        new_dout = Parameter(bprop)
        replace(dout, new_dout)
        bprop.parameters = [new_dout]

    result = app(primops.J, app(prim, *transf_args))
    # outer.output = app(primops.make_tuple, result, Constant(bprop))
    outer.output = app(primops.cons_tuple, result,
                       app(primops.cons_tuple, Constant(bprop), Constant(())))
    return outer


implementations = Registry()
register = implementations.register


def register_bprop(prim):
    def deco(fn):
        fn2 = transform_bprop(prim, fn)
        return register(prim)(fn2)
    return deco


def register_grad(prim):
    def deco(fn):
        fn2 = parse(fn)
        return register(prim)(fn2)
    return deco


@register_bprop(primops.add)
def bprop_add(x, y, dz):
    return (dz, dz)


@register_bprop(primops.sub)
def bprop_sub(x, y, dz):
    return (dz, -dz)


@register_bprop(primops.mul)
def bprop_mul(x, y, dz):
    return (dz * y, dz * x)


@register_bprop(primops.div)
def bprop_div(x, y, dz):
    return (dz / y, -dz * x / (y * y))


@register_bprop(primops.uadd)
def bprop_uadd(x, dz):
    return (dz,)


@register_bprop(primops.usub)
def bprop_usub(x, dz):
    return (-dz,)


@register_bprop(primops.gt)
def bprop_gt(x, y, dz):
    return (zeros_like(x), zeros_like(y))


@register_bprop(primops.lt)
def bprop_lt(x, y, dz):
    return (zeros_like(x), zeros_like(y))


@register_bprop(primops.cons_tuple)
def bprop_cons_tuple(_head, _tail, dz):
    return (head(dz), tail(dz))


@register_bprop(primops.head)
def bprop_head(tup, dz):
    return (cons_tuple(dz, zeros_like(tail(tup))),)


@register_bprop(primops.tail)
def bprop_tail(tup, dz):
    return (cons_tuple(zeros_like(head(tup)), dz),)


@register_bprop(primops.getitem)
def bprop_getitem(data, idx, dz):
    return (setitem(data, idx, dz), zeros_like(idx))


@register_bprop(primops.J)
def bprop_J(x, dz):
    return (Jinv(dz),)


@register_bprop(primops.Jinv)
def bprop_Jinv(x, dz):
    return (J(dz),)


@register_grad(primops.if_)
def bprop_if_(c, tb, fb):
    zeros_like  # Currently required for parser to see it in bprop()

    if Jinv(c):
        rval, branch_bprop = tb()
    else:
        rval, branch_bprop = fb()

    def bprop(dout):
        if Jinv(c):
            return (), zeros_like(c), branch_bprop(dout)[0], zeros_like(fb)
        else:
            return (), zeros_like(c), zeros_like(tb), branch_bprop(dout)[0]

    return rval, bprop