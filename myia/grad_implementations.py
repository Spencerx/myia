
from myia.utils import Registry
from myia.api import parse
from myia.info import NamedDebugInfo, About
from myia.anf_ir import Graph, Apply, Constant, Parameter
from myia import primops
from myia.anf_ir_utils import replace


class GradRegistry(Registry):
    def __setitem__(self, prim, fn):
        info = NamedDebugInfo(prim=prim, name=prim.name)

        bprop = parse(fn)
        bprop.debug.name = None
        bprop.debug.about = About(info, 'grad_bprop')
        empty_tuple = Apply([Constant(primops.make_tuple)], bprop)
        bprop.output.inputs.insert(1, empty_tuple)

        *args, dout = bprop.parameters

        with About(info, 'grad_fw'):
            outer = Graph()

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
        outer.output = app(primops.make_tuple, result, Constant(bprop))

        super().__setitem__(prim, outer)


implementations = GradRegistry()
register = implementations.register


@register(primops.add)
def bprop_add(x, y, dz):
    return (dz, dz)


@register(primops.sub)
def bprop_sub(x, y, dz):
    return (dz, -dz)


@register(primops.mul)
def bprop_mul(x, y, dz):
    return (dz * y, dz * x)


@register(primops.div)
def bprop_div(x, y, dz):
    return (dz / y, -dz * x / (y * y))


@register(primops.uadd)
def bprop_uadd(x, dz):
    return (dz,)


@register(primops.usub)
def bprop_usub(x, dz):
    return (-dz,)

