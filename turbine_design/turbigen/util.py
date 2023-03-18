"""Miscellaneous utility functions that don't fit anywhere else."""
from . import compflow as cf
import numpy as np
import os
from collections import namedtuple


def make_namedtuple_with_docstrings(class_name, class_doc, field_doc):
    nt = namedtuple(class_name, field_doc.keys())
    # Apply documentation, only works for Python 3
    try:
        nt.__doc__ = class_doc
        for fi, vi in field_doc.items():
            getattr(nt, fi).__doc__ = vi
    except AttributeError:
        pass
    return nt


def merge_dicts(a, b):
    c = a.copy()
    c.update(b)
    return c


def boundary_layer_Po_Poinf(spf, delta, Mainf, ga, verbose=False):
    """Return stagnation pressure ratio in a boundary layer."""

    expon = 1.0 / 7.0
    d99 = delta * 0.99

    # Specify velocity
    V_Vinf = np.ones_like(spf)
    V_Vinf[spf < d99] = (spf[spf < d99] / delta) ** expon
    V_Vinf[spf > (1.0 - d99)] = (
        (1.0 - spf[spf > (1.0 - d99)]) / delta
    ) ** expon

    # Evaluate thicknesses and shape factor
    del_star = np.trapz(1.0 - V_Vinf[spf < d99], spf[spf < d99])
    theta = np.trapz(
        (1.0 - V_Vinf[spf < d99]) * V_Vinf[spf < d99], spf[spf < d99]
    )
    H = del_star / theta

    if verbose:
        print("H", H)
        print("del_star", del_star)
        print("theta", theta)

    # Get Mach number
    V_cpTo_inf = cf.V_cpTo_from_Ma(Mainf, ga)
    Po_Pinf = cf.Po_P_from_Ma(Mainf, ga)
    V_cpTo = V_cpTo_inf * V_Vinf
    Ma = cf.Ma_from_V_cpTo(V_cpTo, ga)

    # Transform to stagnation pressure ratio
    Po_P = cf.Po_P_from_Ma(Ma, ga)
    Po_Poinf = Po_P / Po_Pinf

    return Po_Poinf


def make_rundir(base_dir):
    """Inside base_dir, make new work dir in random integer format."""
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    # Make a working directory with unique filename
    case_str = runid_to_str(np.random.randint(0, 1e12))
    workdir = os.path.join(base_dir, case_str)
    os.mkdir(workdir)
    # Return the working directory so that we can save input files there
    return workdir


def runid_to_str(runid):
    return "%012d" % runid


class Frozen:
    """Inherit from this class to disable of new attributes using _freeze().

    https://stackoverflow.com/a/3603824"""

    __isfrozen = False

    def __setattr__(self, key, value):
        if self.__isfrozen and key not in dir(self):
            raise TypeError("%r attributes are frozen" % self)
            object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True

    def _unfreeze(self):
        self.__isfrozen = False
