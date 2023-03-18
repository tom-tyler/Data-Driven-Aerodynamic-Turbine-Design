"""Functions to add cooling to a Turbostream grid."""

import json, os
import numpy as np
from turbigen import compflow
from . import grid

class Locations():
    HUB_RIM = 0
    SHROUD_RIM = 1
    PRESSURE_SIDE = 2
    SUCTION_SIDE = 3
    HUB_PLATFORM = 4
    SHROUD_PLATFORM = 5
    TRAILING_EDGE = 6



class CoolingPatch:
    def __init__(self, fc, TR, Ma, Alpha, Beta, location, surface_fraction):
        """Non-dimensional data for a single cooling patch."""

        self.fc = fc
        self.TR = TR
        self.Ma = Ma
        self.Alpha = Alpha
        self.Beta = Beta

        try:
            self.location = getattr(Locations,location)
        except TypeError:
            self.location = location
        self.surface_fraction = surface_fraction

    def to_dict(self):
        d = {
            "fc": self.fc,
            "TR": self.TR,
            "Ma": self.Ma,
            "Alpha": self.Alpha,
            "Beta": self.Beta,
            "location": self.location,
            "surface_fraction": self.surface_fraction,
        }
        return d


def _ist_from_zeta(g, irow, zeta_target):

    c = g.cut_blade_surfs()[irow]
    jmid = g.span_fraction_index(0.5, irow)
    zeta = c.zeta[:, jmid]
    zeta[zeta < 0.0] = -zeta[zeta < 0.0] / np.min(zeta[zeta < 0.0])
    zeta[zeta > 0.0] = zeta[zeta > 0.0] / np.max(zeta[zeta > 0.0])
    ind = np.arange(0.0, len(zeta))
    return np.round(np.interp(zeta_target, zeta, ind)).astype(int)

def sum_coolant_enthalpy(g):
    """Total inflow of enthalpy by cooling patches."""
    cp = g.get_av("cp")
    Ho = 0.
    mdot = 0.
    for bid_cool, pid_cool in zip(*g.find_patches(grid.KIND.cooling)):
        nb = float(g.get_bv("nblade", bid_cool))
        mdotc = g.get_pv("cool_mass", bid_cool, pid_cool)
        Toc = g.get_pv("cool_tstag", bid_cool, pid_cool)
        mdot += mdotc*nb
        Ho += mdotc*cp*Toc*nb
    return Ho, mdot

def weighted_pressure(g, mdot1, Po1):
    """The ideal weighted pressure after Young and Horlock (2006), Eqn (18)."""

    ga = g.get_av("ga")

    # Loop over patches
    Pow_Po1 = 1.
    for bid_cool, pid_cool in zip(*g.find_patches(grid.KIND.cooling)):

        # Get patch variables
        nb = float(g.get_bv("nblade", bid_cool))
        mdotc = nb*g.get_pv("cool_mass", bid_cool, pid_cool)
        Mac = g.get_pv("cool_mach", bid_cool, pid_cool)

        # Roughly average static pressure
        # Should change to a proper area average
        patch = g.cut_patch(bid_cool, pid_cool)
        Pc = np.mean(patch.pstat)
        Poc = compflow.Po_P_from_Ma(Mac, ga) * Pc

        fc = mdotc/mdot1
        expon = fc/(1.+fc)

        Pow_Po1 *= (Poc/Po1)**expon

    return Pow_Po1 * Po1


    return Pow_Po1

def save_json(patches, fname):
    """Write a list of cooling patches to a JSON file."""
    d = {}
    for j, pj in enumerate(patches):
        kj = "row_%d" % j
        d[kj] = {}
        for i, pi in enumerate(pj):
            ki = "patch_%d" % i
            d[kj][ki] = pi.to_dict()

    with open(fname, "w") as f:
        json.dump(d, f, indent=4)


def load_json(fname):
    """Read a list of cooling patches from a JSON file."""
    with open(fname, "r") as f:
        d = json.load(f)
    return [[CoolingPatch(**pi) for pi in pj.values()] for pj in d.values()]


def add_to_grid(stage_patches, g, mdot, Toinf):
    """Add cooling patches to a Turbostream grid."""

    cut_rows = g.cut_rows()

    # Loop over rows
    for i, patches in enumerate(stage_patches):

        nb = float(g.get_bv("nblade", cut_rows[i][0].bid))

        # Loop over patches
        for p in patches:

            if not p.fc:
                continue

            # Choose how to apply based on location
            if p.location == Locations.HUB_RIM:

                c = cut_rows[i][0]
                bid = c.bid
                pid = g.make_patch(
                    kind="cooling",
                    bid=bid,
                    i=(g.ni[bid] - 3, g.ni[bid]),
                    j=(0, 1),
                    k=(0, g.nk[bid]),
                )[1]

            elif p.location == Locations.SHROUD_RIM:

                c = cut_rows[i][0]
                bid = c.bid
                pid = g.make_patch(
                    kind="cooling",
                    bid=bid,
                    i=(g.ni[bid] - 3, g.ni[bid]),
                    j=(g.nj[bid] - 1, g.nj[bid]),
                    k=(0, g.nk[bid]),
                )[1]

            elif p.location == Locations.HUB_PLATFORM:

                bid = g.omesh_bids[i] + 1
                pid = g.make_patch(
                    kind="cooling",
                    bid=bid,
                    i=(0, g.ni[bid]),
                    j=(0, 1),
                    k=(g.nk[bid] - 4, g.nk[bid]),
                )[1]

            elif p.location == Locations.SHROUD_PLATFORM:

                bid = g.omesh_bids[i] + 1

                pid = g.make_patch(
                    kind="cooling",
                    bid=bid,
                    i=(0, g.ni[bid]),
                    j=(g.nj[bid] - 1, g.nj[bid]),
                    k=(g.nk[bid] - 4, g.nk[bid]),
                )[1]

            elif p.location == Locations.PRESSURE_SIDE:

                bid = g.omesh_bids[i]
                ist = _ist_from_zeta(g, i, p.surface_fraction)
                pid = g.make_patch(
                    kind="cooling",
                    bid=bid,
                    i=(ist, ist + 4),
                    j=(0, g.nj[bid]),
                    k=(0, 1),
                )[1]

            elif p.location == Locations.SUCTION_SIDE:

                bid = g.omesh_bids[i]
                ist = _ist_from_zeta(g, i, -p.surface_fraction)
                pid = g.make_patch(
                    kind="cooling",
                    bid=bid,
                    i=(ist - 4, ist),
                    j=(0, g.nj[bid]),
                    k=(0, 1),
                )[1]

            elif p.location == Locations.TRAILING_EDGE:

                bid = g.omesh_bids[i]
                pid = g.make_patch(
                    kind="cooling",
                    bid=bid,
                    i=(g.ni[bid] - 6, g.ni[bid]),
                    j=(0, g.nj[bid]),
                    k=(0, 1),
                )[1]

            # Set boundary conditions on patch
            g.set_pv("cool_mass", bid, pid, p.fc * mdot[i] / nb)
            g.set_pv("cool_mach", bid, pid, p.Ma)
            g.set_pv("cool_tstag", bid, pid, p.TR * Toinf[i])
            g.set_pv("cool_sangle", bid, pid, p.Alpha)
            g.set_pv("cool_xangle", bid, pid, p.Beta)

            # Misc patch variables
            g.set_pv("cool_angle_def", bid, pid, 0)
            g.set_pv("cool_type", bid, pid, 0)
            g.set_pv("cool_frac_area", bid, pid, 1.0)
            g.set_pv("cool_pstag", bid, pid, 16e5)  # Not used by TS


DEFAULT_PATH = os.path.join(os.path.dirname(__file__),'cooling.json')
DEFAULT = load_json(DEFAULT_PATH)
