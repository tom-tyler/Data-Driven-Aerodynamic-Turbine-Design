"""A user-friendly wrapper around the official Turbostream functions."""
import matplotlib
matplotlib.use('agg')
try:
    import ts_tstream_default, ts_tstream_reader, ts_tstream_load_balance
    from ts_tstream_grid import TstreamGrid
    from ts.ts_tstream_cut import TstreamStructuredCut
    import ts.ts_tstream_type as TYPE
    import ts_tstream_patch_kind as KIND
except ImportError:
    raise Exception(
        (
            """
Turbostream modules not found. You need to setup the environment first.

On a login node (login-e-*) run:
    source /usr/local/software/turbostream/ts3610/bashrc_module_ts3610

On a compute node (gpu-q-*) run:
    source /usr/local/software/turbostream/ts3610_a100/bashrc_module_ts3610_a100"""
        )
    )

from . import ts_autogrid_reader
from turbigen import compflow_native as compflow
from turbigen import average
import numpy as np
from scipy.interpolate import interp1d, interpn
import os, sys

COORD_NAMES = ["x", "r", "rt"]


class suppress_print:
    """A context manager that temporarily sets STDOUT to /dev/null."""

    def __enter__(self):
        self.orig_out = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stdout = self.orig_out


def _squeeze_list(x):
    if len(x) == 1:
        return x[0]
    else:
        return x


def _calculate_zeta(x, rt, normalise=False):
    # Cumsum of steps along i coordinate
    dx = np.diff(x, n=1, axis=0)
    drt = np.diff(rt, n=1, axis=0)
    dzeta = np.sqrt(dx ** 2.0 + drt ** 2.0)
    zeta = np.cumsum(dzeta, axis=0)
    zeta0 = np.zeros((1, dzeta.shape[1]))
    zeta = np.insert(zeta, 0, zeta0, axis=0)

    # Make leading edge zero at all radial locations
    ile = np.expand_dims(np.argmin(x, axis=0), 0)
    zeta_le = np.take_along_axis(zeta, ile, axis=0)
    zeta -= zeta_le

    # Normalise trailing edges to unity
    if normalise:
        zeta_max = np.tile(
            zeta.max(axis=0, keepdims=True), (zeta.shape[0], 1)
        )
        zeta_min = np.tile(
            zeta.min(axis=0, keepdims=True), (zeta.shape[0], 1)
        )
        ip = zeta > 0.0
        im = zeta <= 0.0
        zeta[ip] = zeta[ip] / zeta_max[ip]
        zeta[im] = -zeta[im] / zeta_min[im]

    return zeta


def _join_cuts(cuts, axis, flip):
    cut_out = cuts[0]
    for vn in dir(cuts[0]):
        v0 = getattr(cuts[0], vn)
        v1 = getattr(cuts[1], vn)
        if np.ndim(v0) == 2:
            if flip:
                v0 = np.flip(v0, axis=axis)

            # Remove duplicate point
            v1m = np.moveaxis(v1, axis, 0)
            v1m = np.delete(v1m, 0, 0)
            v1 = np.moveaxis(v1m, 0, axis)

            vc = np.concatenate((v0, v1), axis=axis)
            setattr(cut_out, vn, vc)

    return cut_out


# Names of valid block properties
BP_NAMES = [
    "x",
    "r",
    "rt",
    "mwall",
    "phi",
    "ro",
    "rovx",
    "rovr",
    "rorvt",
    "roe",
    "tlength",
    "trans_dyn_vis",
    "xlength",
    "yplus",
    "tdamp",
    "fac_st0",
    "dwallsq",
    "xlength",
    # Begin useless BPs
    "rok",
    "misc",
    "ent",
    "pstag",
    "asound",
    "roym0",
    "tstag_rel",
    "tstat",
    "hstag",
    "tstag",
    "roomega",
]

# Names of valid patch properties
PP_NAMES = ["pstag", "tstag", "yaw", "pitch", "fsturb_mul", "pout"]

# Type (float or int) of all application variables
AV_TYPES = {
    "cp": TYPE.float,
    "ga": TYPE.float,
    "cfl": TYPE.float,
    "prandtl": TYPE.float,
    "viscosity": TYPE.float,
    "sfin": TYPE.float,
    "dampin": TYPE.float,
    "rfvis": TYPE.float,
    "nstep": TYPE.int,
    "nchange": TYPE.int,
    "poisson_sfin": TYPE.float,
    "poisson_cfl": TYPE.float,
    "poisson_nstep": TYPE.int,
    "poisson_restart": TYPE.int,
    "restart": TYPE.int,
    "ilos": TYPE.int,
    "nlos": TYPE.int,
    "dts": TYPE.int,
    "ncycle": TYPE.int,
    "frequency": TYPE.float,
    "nstep_cycle": TYPE.int,
    "dts_conv": TYPE.float,
    "facsafe": TYPE.float,
    "nstep_inner": TYPE.int,
    "nstep_save": TYPE.int,
    "nstep_save_start": TYPE.int,
    "nstep_save_probe": TYPE.int,
    "nstep_save_start_probe": TYPE.int,
    "rfmix": TYPE.float,
    "facsecin": TYPE.float,
    "viscosity_law": TYPE.int,
    "wall_law": TYPE.int,
    "write_yplus": TYPE.int,
    "write_force": TYPE.int,
    "if_ale": TYPE.int,
    "adaptive_smoothing": TYPE.int,
    "ifgas": TYPE.int,
    "tref": TYPE.float,
    "rg_cp0": TYPE.float,
    "rg_cp1": TYPE.float,
    "rg_cp2": TYPE.float,
    "rg_cp3": TYPE.float,
    "rg_cp4": TYPE.float,
    "rg_cp5": TYPE.float,
    "rg_rgas": TYPE.float,
    "nspecies": TYPE.int,
    "sf_scalar": TYPE.float,
    "schmidt_0": TYPE.float,
    "schmidt_1": TYPE.float,
    "cp0_0": TYPE.float,
    "cp1_0": TYPE.float,
    "cp2_0": TYPE.float,
    "cp3_0": TYPE.float,
    "cp4_0": TYPE.float,
    "cp5_0": TYPE.float,
    "rgas_0": TYPE.float,
    "cp0_1": TYPE.float,
    "cp1_1": TYPE.float,
    "cp2_1": TYPE.float,
    "cp3_1": TYPE.float,
    "cp4_1": TYPE.float,
    "cp5_1": TYPE.float,
    "rgas_1": TYPE.float,
    "pref": TYPE.float,
    "fac_stmix": TYPE.float,
    "fac_st0": TYPE.float,
    "fac_st1": TYPE.float,
    "fac_st2": TYPE.float,
    "fac_st3": TYPE.float,
    "fac_sa_step": TYPE.float,
    "fac_sa_smth": TYPE.float,
    "turb_vis_damp": TYPE.float,
    "fac_wall": TYPE.float,
    "if_no_mg": TYPE.int,
    "turbvis_lim": TYPE.float,
    "nomatch_int": TYPE.int,
    "poisson_nsmooth": TYPE.int,
    "write_tdamp": TYPE.int,
    "write_egen": TYPE.int,
    "viscosity_a1": TYPE.float,
    "viscosity_a2": TYPE.float,
    "viscosity_a3": TYPE.float,
    "viscosity_a4": TYPE.float,
    "viscosity_a5": TYPE.float,
    "ifsuperfac": TYPE.int,
    "poisson_limit": TYPE.int,
    "cfl_ko": TYPE.float,
    "cfl_st_ko": TYPE.float,
    "cfl_en_ko": TYPE.float,
    "sfin_sa": TYPE.float,
    "sfin_ko": TYPE.float,
    "ko_dist": TYPE.float,
    "ko_restart": TYPE.int,
    "fac_st0_option": TYPE.int,
    "sa_helicity_option": TYPE.int,
    "sa_ch1": TYPE.float,
    "sa_ch2": TYPE.float,
    "precon": TYPE.int,
    "smooth_scale_dts_option": TYPE.int,
    "smooth_scale_precon_option": TYPE.int,
    "smooth_scale_directional_option": TYPE.int,
    "use_temperature_sensor": TYPE.int,
}

# Type (float or int) of all block variables
BV_TYPES = {
    "rpm": TYPE.float,
    "rpmi1": TYPE.float,
    "rpmi2": TYPE.float,
    "rpmj1": TYPE.float,
    "rpmj2": TYPE.float,
    "rpmk1": TYPE.float,
    "rpmk2": TYPE.float,
    "fmgrid": TYPE.float,
    "poisson_fmgrid": TYPE.float,
    "xllim": TYPE.float,
    "vgridin": TYPE.float,
    "pstatin": TYPE.float,
    "tstagin": TYPE.float,
    "vgridout": TYPE.float,
    "pstatout": TYPE.float,
    "tstagout": TYPE.float,
    "ftype": TYPE.int,
    "nblade": TYPE.int,
    "fblade": TYPE.float,
    "fracann": TYPE.float,
    "sfin_mul": TYPE.float,
    "facsecin_mul": TYPE.float,
    "dampin_mul": TYPE.float,
    "fsturb": TYPE.float,
    "xllim_free": TYPE.float,  # Not in official defaults
    "free_turb": TYPE.float,  # Not in official defaults
    "nimixl": TYPE.int,
    "srough_i0": TYPE.float,
    "srough_i1": TYPE.float,
    "srough_j0": TYPE.float,
    "srough_j1": TYPE.float,
    "srough_k0": TYPE.float,
    "srough_k1": TYPE.float,
    "itrans": TYPE.int,
    "itrans_j1_st": TYPE.int,
    "itrans_j2_st": TYPE.int,
    "itrans_k1_st": TYPE.int,
    "itrans_k2_st": TYPE.int,
    "itrans_j1_en": TYPE.int,
    "itrans_j2_en": TYPE.int,
    "itrans_k1_en": TYPE.int,
    "itrans_k2_en": TYPE.int,
    "itrans_j1_frac": TYPE.float,
    "itrans_j2_frac": TYPE.float,
    "itrans_k1_frac": TYPE.float,
    "itrans_k2_frac": TYPE.float,
    "jtrans": TYPE.int,
    "jtrans_i1_st": TYPE.int,
    "jtrans_i2_st": TYPE.int,
    "jtrans_k1_st": TYPE.int,
    "jtrans_k2_st": TYPE.int,
    "jtrans_i1_en": TYPE.int,
    "jtrans_i2_en": TYPE.int,
    "jtrans_k1_en": TYPE.int,
    "jtrans_k2_en": TYPE.int,
    "jtrans_i1_frac": TYPE.float,
    "jtrans_i2_frac": TYPE.float,
    "jtrans_k1_frac": TYPE.float,
    "jtrans_k2_frac": TYPE.float,
    "ktrans": TYPE.int,
    "ktrans_i1_st": TYPE.int,
    "ktrans_i2_st": TYPE.int,
    "ktrans_j1_st": TYPE.int,
    "ktrans_j2_st": TYPE.int,
    "ktrans_i1_en": TYPE.int,
    "ktrans_i2_en": TYPE.int,
    "ktrans_j1_en": TYPE.int,
    "ktrans_j2_en": TYPE.int,
    "ktrans_i1_frac": TYPE.float,  # Typo in official defaults
    "ktrans_i2_frac": TYPE.float,  # Typo in official defaults
    "ktrans_j1_frac": TYPE.float,  # Typo in official defaults
    "ktrans_j2_frac": TYPE.float,  # Typo in official defaults
    "superfac": TYPE.float,
    "fac_st0": TYPE.float,
    "ndup_phaselag": TYPE.int,
    "turb_intensity": TYPE.float,
    "fl_ibpa": TYPE.float,
}

# Type (float or int) of all patch variables
PV_TYPES = {
    "rfin": TYPE.float,
    "sfinlet": TYPE.float,
    "pout": TYPE.float,
    "ipout": TYPE.int,
    "throttle_type": TYPE.int,
    "throttle_target": TYPE.float,
    "throttle_k0": TYPE.float,
    "throttle_k1": TYPE.float,
    "throttle_k2": TYPE.float,
    "pout_en": TYPE.float,
    "pout_st": TYPE.float,
    "pout_nchange": TYPE.int,
    "pstagfixed": TYPE.float,
    "tstagfixed": TYPE.float,
    "vxfixed": TYPE.float,
    "vrfixed": TYPE.float,
    "vtfixed": TYPE.float,
    "cool_type": TYPE.int,
    "cool_mass": TYPE.float,
    "cool_pstag": TYPE.float,
    "cool_tstag": TYPE.float,
    "cool_mach": TYPE.float,
    "cool_frac_area": TYPE.float,
    "cool_sangle": TYPE.float,
    "cool_xangle": TYPE.float,
    "bleed_flow": TYPE.float,
    "probe_append": TYPE.int,
    "slide_nxbid": TYPE.int,
    "slide_nxpid": TYPE.int,
    "shroud_sealgap": TYPE.float,
    "shroud_nseal": TYPE.int,
    "shroud_cfshroud": TYPE.float,
    "shroud_cfcasing": TYPE.float,
    "shroud_wshroud": TYPE.float,
    "shroud_wcase": TYPE.float,
    "shroud_pitchin": TYPE.float,
    "shroud_dir": TYPE.int,
    "pstag_free": TYPE.float,
    "tstag_free": TYPE.float,
    "pstat_free": TYPE.float,
    "sf_free": TYPE.float,
    "group": TYPE.int,
    "porous_fac_loss": TYPE.float,
    "porous_rf": TYPE.float,
    # Questionable
    "sep_drag": TYPE.float,
    "rg_cp0": TYPE.float,
    "rg_cp1": TYPE.float,
    "rg_cp2": TYPE.float,
    "rg_cp3": TYPE.float,
    "rg_cp4": TYPE.float,
    "rg_cp5": TYPE.float,
    "rg_rgas": TYPE.float,
    "fthrottle": TYPE.float,
    "cool_angle_def": TYPE.int,
    "phaselag_periodic_sign": TYPE.float,
}


def _check_av_name(name):
    if name not in AV_TYPES:
        raise Exception(
            "'%s' is not a valid application variable" % name
        )


def _check_bp_name(name):
    if name not in BP_NAMES:
        raise Exception("'%s' is not a valid block property" % name)


def _check_bv_name(name):
    if name not in BV_TYPES:
        raise Exception("'%s' is not a valid block variable" % name)


def _check_pp_name(name):
    if name not in PP_NAMES:
        raise Exception("'%s' is not a valid patch property" % name)


def _check_pv_name(name):
    if name not in PV_TYPES:
        raise Exception("'%s' is not a valid patch variable" % name)


def _convert_to_int(x):
    """Coerce the input to a scalar integer."""
    # Check for remainder when dividing by one
    # Error if we have a fractional part left over
    if x % 1:
        raise Exception("%f is not an integer" % x)
    return int(x)


def _convert_to_float(x):
    """Coerce the input to a float."""
    return float(x)


def _is_scalar(x):
    """Return true for length-1 1D arrays or scalars or 0D arrays."""
    return np.shape(x) in ((), (1,))


def _choose_topology(g):
    """Take a vanilla Turbostream grid and convert to OH or H grid topology."""
    # Convert to improved grid
    Grid.convert(g)
    # Pick grid topology
    if g.is_h_mesh:
        HGrid.convert(g)
    else:
        OHGrid.convert(g)


def read_autogrid(bcs_file, g_file):
    """Load grid from Autogrid boundary condition and geometry files."""

    # Error if files do not exist
    if not os.path.exists(bcs_file):
        raise Exception("bcs file %s not found" % bcs_file)
    if not os.path.exists(g_file):
        raise Exception("g file %s not found" % g_file)

    # Load using the vanilla TS Autogrid reader
    agr = ts_autogrid_reader.AutogridReader()
    with suppress_print():
        g = agr.read(bcs_file, g_file)

    # Perform conversion
    _choose_topology(g)

    return g


def read_hdf5(fname):
    """Load grid from Turbostream HDF5 file."""

    # Error if file does not exist
    if not os.path.exists(fname):
        raise Exception("HDF5 file does not exist %s" % fname)

    # Load using vanilla Turbostream reader
    tsr = ts_tstream_reader.TstreamReader()
    with suppress_print():
        g = tsr.read(fname)

    _choose_topology(g)

    return g


class Cut:
    """Collect flow variables on a cut as attributes."""

    def __init__(
        self, g, bid, ist, ien, jst, jen, kst, ken, squeeze=True
    ):

        self.bid = bid
        self.ist = ist
        self.ien = ien
        self.jst = jst
        self.jen = jen
        self.kst = kst
        self.ken = ken
        self.ni = ien - ist
        self.nj = jen - jst
        self.nk = ken - kst
        self.ga = g.get_av("ga")
        self.cp = g.get_av("cp")
        self.rpm = g.get_bv("rpm", bid)
        self.nblade = g.get_bv("nblade", bid)
        self.Omega = self.rpm / 60.0 * 2.0 * np.pi
        self.rgas = self.cp * (self.ga - 1.0) / self.ga

        # Arbitrary reference pressures for entropy calculation
        # Only changes in entropy are physically meaningful
        self.pref = 1e5
        self.tref = 300.0

        # Deal with end indices inclusively
        ni, nj, nk = g.nijk[g.bids.index(bid)]

        if ien < 0:
            ien = ni + ien + 1
        if jen < 0:
            jen = nj + jen + 1
        if ken < 0:
            ken = nk + ken + 1

        if ist < 0:
            ist = ni + ist + 1
        if jst < 0:
            jst = nj + jst + 1
        if kst < 0:
            kst = nk + kst + 1

        # Always get coordinates
        self.x = g.get_bp("x", bid)[ist:ien, jst:jen, kst:ken]
        self.r = g.get_bp("r", bid)[ist:ien, jst:jen, kst:ken]
        self.rt = g.get_bp("rt", bid)[ist:ien, jst:jen, kst:ken]

        # Fetch flow solution if it exists
        # get_bp returns NaNs of correct shape otherwise
        self.ro = g.get_bp("ro", bid, raise_missing=False)[
            ist:ien, jst:jen, kst:ken
        ]
        self.rovx = g.get_bp("rovx", bid, raise_missing=False)[
            ist:ien, jst:jen, kst:ken
        ]
        self.rovr = g.get_bp("rovr", bid, raise_missing=False)[
            ist:ien, jst:jen, kst:ken
        ]
        self.rorvt = g.get_bp("rorvt", bid, raise_missing=False)[
            ist:ien, jst:jen, kst:ken
        ]
        self.roe = g.get_bp("roe", bid, raise_missing=False)[
            ist:ien, jst:jen, kst:ken
        ]

        # Reduce to a 2D array if requested
        if squeeze:

            self.x = np.squeeze(self.x)
            self.r = np.squeeze(self.r)
            self.rt = np.squeeze(self.rt)

            self.ro = np.squeeze(self.ro)
            self.rovx = np.squeeze(self.rovx)
            self.rovr = np.squeeze(self.rovr)
            self.rorvt = np.squeeze(self.rorvt)
            self.roe = np.squeeze(self.roe)

        self.t = (
            self.rt.astype(np.float64) / self.r.astype(np.float64)
        ).astype(np.float32)

        # Divide out density
        self.vx = self.rovx / self.ro
        self.vr = self.rovr / self.ro
        self.vt = self.rorvt / self.ro / self.r
        e = self.roe / self.ro

        # Velocities
        self.vsq = self.vx ** 2.0 + self.vr ** 2.0 + self.vt ** 2.0
        self.U = self.r * self.Omega
        self.vt_rel = self.vt - self.U
        self.vsq_rel = (
            self.vx ** 2.0 + self.vr ** 2.0 + self.vt_rel ** 2.0
        )

        # Pressure and temperature
        cv = self.cp / self.ga
        self.tstat = (e - 0.5 * self.vsq) / cv
        self.pstat = self.ro * self.rgas * self.tstat

        # Mach
        self.mach = np.sqrt(self.vsq / self.ga / self.rgas / self.tstat)
        self.mach_rel = np.sqrt(
            self.vsq_rel / self.ga / self.rgas / self.tstat
        )

        # Stagnation pressures and temperatures
        self.pstag = self.pstat * compflow.Po_P_from_Ma(
            self.mach, self.ga
        )
        self.tstag = self.tstat * compflow.To_T_from_Ma(
            self.mach, self.ga
        )
        self.pstag_rel = self.pstat * compflow.Po_P_from_Ma(
            self.mach_rel, self.ga
        )
        self.tstag_rel = self.tstat * compflow.To_T_from_Ma(
            self.mach_rel, self.ga
        )

        # Span fraction
        if (self.r.ptp(axis=1) > 0.0).all():
            self.spf = (
                self.r - self.r.min(axis=1, keepdims=True)
            ) / self.r.ptp(axis=1, keepdims=True)
        else:
            self.spf = np.ones_like(self.r) * np.nan

        # Cartesian coordinates
        self.y = self.r * np.sin(self.t)
        self.z = self.r * np.cos(self.t)

    def mix_out(self):
        """Take a structured cuts and mix out the flow at constant area."""

        # Gas properties
        cp = self.cp
        ga = self.ga
        rgas = cp * (ga - 1.0) / ga
        Omega = self.rpm / 60.0 * 2.0 * np.pi

        # Do the mixing
        (
            r_mix,
            ro_mix,
            rovx_mix,
            rovr_mix,
            rorvt_mix,
            roe_mix,
            Ax,
        ) = average.mix_out(
            self.x,
            self.r,
            self.rt,
            self.ro,
            self.rovx,
            self.rovr,
            self.rorvt,
            self.roe,
            ga,
            rgas,
            Omega,
        )

        # Secondary mixed vars
        (
            vx_mix,
            vr_mix,
            vt_mix,
            P_mix,
            T_mix,
        ) = average.primary_to_secondary(
            r_mix,
            ro_mix,
            rovx_mix,
            rovr_mix,
            rorvt_mix,
            roe_mix,
            ga,
            rgas,
        )

        # Max a new cut with the mixed out flow
        cut_out = TstreamStructuredCut()

        cut_out.pref = self.pref
        cut_out.tref = self.tref
        cut_out.ni = 1
        cut_out.nj = 1
        cut_out.nk = 1
        cut_out.ga = ga
        cut_out.cp = cp
        cut_out.ifgas = 0
        cut_out.write_egen = 0
        cut_out.nblade = self.nblade

        cut_out.rpm = self.rpm
        cut_out.x = np.mean(self.x)
        cut_out.r = r_mix
        cut_out.rt = np.mean(self.rt)
        cut_out.ro = ro_mix
        cut_out.rovx = rovx_mix
        cut_out.rovr = rovr_mix
        cut_out.rorvt = rorvt_mix
        cut_out.roe = roe_mix
        cut_out.t = cut_out.rt / cut_out.r

        cut_out.tstat = T_mix
        cut_out.pstat = P_mix

        cut_out.vx = vx_mix
        cut_out.vr = vr_mix
        cut_out.vt = vt_mix
        cut_out.vabs = np.sqrt(
            vx_mix ** 2.0 + vr_mix ** 2.0 + vt_mix ** 2.0
        )
        cut_out.U = r_mix * Omega
        cut_out.vt_rel = vt_mix - cut_out.U

        cut_out.vabs_rel = np.sqrt(
            cut_out.vx ** 2.0
            + cut_out.vr ** 2.0
            + cut_out.vt_rel ** 2.0
        )
        cut_out.mach_rel = cut_out.vabs_rel / np.sqrt(
            ga * rgas * cut_out.tstat
        )

        cut_out.mach = cut_out.vabs / np.sqrt(ga * rgas * cut_out.tstat)
        cut_out.pstag = (
            compflow.Po_P_from_Ma(cut_out.mach, ga) * cut_out.pstat
        )
        cut_out.pstag_rel = (
            compflow.Po_P_from_Ma(cut_out.mach_rel, ga) * cut_out.pstat
        )
        cut_out.tstag = (
            compflow.To_T_from_Ma(cut_out.mach, ga) * cut_out.tstat
        )

        cut_out.yaw = np.degrees(np.arctan2(cut_out.vt, cut_out.vx))
        cut_out.yaw_rel = np.degrees(
            np.arctan2(cut_out.vt_rel, cut_out.vx)
        )

        cut_out.entropy = cp * np.log(
            cut_out.tstat / cut_out.tref
        ) - rgas * np.log(cut_out.pstat / cut_out.pref)

        cut_out.Ax = Ax
        cut_out.mdot = rovx_mix * Ax * cut_out.nblade

        # Cartesian coordinates
        cut_out.y = cut_out.r * np.sin(cut_out.t)
        cut_out.z = cut_out.r * np.cos(cut_out.t)

        return cut_out


class Grid(TstreamGrid):
    """Add new methods and improve existing Turbostream grid methods."""

    # The official Tstream grid is an "old-style" class and does not support
    # super() to call its methods in this descendant class. So we call them as
    # "unbound methods" on the TstreamGrid class providing an explicit self as
    # first argument.

    # We cannot make this class an new-style class (inheriting from object as
    # well) because then we cannot mutate an existing TstreamGrid instance into
    # one of the improved grids.

    @classmethod
    def convert(cls, g):
        """Mutate a vanilla grid object into an improved grid object."""
        # https://stackoverflow.com/questions/990758/reclassing-an-instance-in-python
        g.__class__ = cls

    #
    # Properties for useful block ids and cuts
    #

    @property
    def is_h_mesh(self):
        return self.nrow == self.nblock

    @property
    def nijk(self):
        nijk = []
        for bid in self.bids:
            blk = self.get_block(bid)
            nijk.append((blk.ni, blk.nj, blk.nk))
        return nijk

    @property
    def ni(self):
        return [nijk[0] for nijk in self.nijk]

    @property
    def nj(self):
        return [nijk[1] for nijk in self.nijk]

    @property
    def nk(self):
        return [nijk[2] for nijk in self.nijk]

    @property
    def bids(self):
        return self.get_block_ids()

    @property
    def nblock(self):
        return self.get_nb()

    @property
    def nrow(self):
        return len(self.row_bids)

    @property
    def row_bids(self):
        """Split blocks into rows by location of mixing planes."""

        xmix = np.array(
            [
                self.cut_patch(bid, pid).x.mean()
                for bid, pid in zip(*self.find_patches(KIND.mixing))
            ]
        )
        nrow = len(xmix) / 2 + 1

        if nrow == 1:
            return [
                list(self.bids),
            ]

        xblock = np.array(
            [self.get_bp("x", bid).mean() for bid in self.bids]
        )
        bid_all = np.array(self.bids)
        if nrow == 2:
            xmix = xmix.mean()
            return [
                bid_all[xblock <= xmix].tolist(),
                bid_all[xblock > xmix].tolist(),
            ]
        else:
            raise NotImplementedError("Not implemented multi-stage")

    def pids(self, bid):
        return self.get_patch_ids(bid)

    def cut_patch_kind(self, kind, offset=None):
        return _squeeze_list(
            [
                self.cut_patch(bid, pid, offset=offset)
                for bid, pid in zip(*self.find_patches(kind))
            ]
        )

    def cut_inlet(self):
        return self.cut_patch_kind(KIND.inlet)

    def cut_outlet(self):
        return self.cut_patch_kind(KIND.outlet)

    def cut_mixing(self):
        """Nested tuple of cuts on both sides of mixing plane, for each row."""
        cuts = self.cut_patch_kind(KIND.mixing, offset=1)
        cuts_row = []
        for i in range(self.nrow - 1):
            i2 = 2 * i
            cuts_row.append(cuts[i2 : (i2 + 2)])
        return _squeeze_list(cuts_row)

    def cut_rows(self):
        """Get inlet and outlet cuts for all rows."""
        if self.nrow == 1:
            return [self.cut_patch_kind(), self.cut_outlet()]
        elif self.nrow == 2:
            cut_mix = self.cut_mixing()
            return [
                [self.cut_inlet(), cut_mix[0]],
                [cut_mix[1], self.cut_outlet()],
            ]

    def unstructured_block_coords(self, bid):
        """Take (ni,nj,nk) x, r, rt coordinates and assemble (ni*nj*nk, 3)."""
        return np.stack(
            [self.get_bp(v, bid).reshape(-1) for v in COORD_NAMES],
            axis=-1,
        )

    def restructure_block_coords(self, bid, xrrt):
        """Take (ni*nj*nk, 3) block coords and put back to (ni,nj,nk) x, r, rt."""
        for c, v in zip(xrrt.T, COORD_NAMES):
            self.set_bp(v, bid, c.reshape(self.nijk[bid]))

    @property
    def cx(self):
        return [
            np.mean(cut.x.ptp(axis=0)) for cut in self.cut_blade_surfs()
        ]

    def span_fraction_index(self, spf, row_index=0):
        cut = self.cut_blade_surfs()[row_index]
        r = np.mean(cut.r, 0)
        rnorm = (r - r.min()) / r.ptp()
        try:
            return [np.argmin(np.abs(rnorm - spfi)) for spfi in spf]
        except (ValueError, TypeError):
            return np.argmin(np.abs(rnorm - spf))

    #
    # Setting and getting
    #

    def _check_bid(self, bid):
        """Raise a helpful error if requested block id does not exist."""
        if not bid in self.bids:
            raise Exception(
                "Invalid bid='%s' requested, should be int <= %d"
                % (str(bid), self.nblock - 1)
            )

    def _check_pid(self, bid, pid):
        """Raise a helpful error if requested patch id does not exist."""

        self._check_bid(bid)

        if not pid in self.pids(bid):
            raise Exception(
                "Invalid pid=%d requested on bid=%d, should be int <= %d"
                % (pid, bid, len(self.pids(bid)))
            )

    def set_av(self, name, val):
        """Set an application variable, handling types automatically."""

        _check_av_name(name)

        try:
            var_type = AV_TYPES[name]
            if var_type == TYPE.int:
                var_val = _convert_to_int(val)
            else:
                var_val = _convert_to_float(val)

            TstreamGrid.set_av(self, name, var_type, var_val)

        except KeyError:
            raise Exception("Invalid application variable: %s" % name)

    def set_bv(self, name, val, bid=None):
        """Set a block variable, handling types automatically."""

        _check_bv_name(name)

        try:
            var_type = BV_TYPES[name]
            if var_type == TYPE.int:
                var_val = _convert_to_int(val)
            else:
                var_val = _convert_to_float(val)

            if bid is None:
                for b in self.bids:
                    TstreamGrid.set_bv(self, name, var_type, b, var_val)
            else:
                try:
                    for b in bid:
                        self._check_bid(b)
                        TstreamGrid.set_bv(
                            self, name, var_type, b, var_val
                        )
                except TypeError:
                    TstreamGrid.set_bv(
                        self, name, var_type, bid, var_val
                    )

        except KeyError:
            raise Exception("Invalid block variable: %s" % name)

    def set_pp(self, name, bid, pid, val):
        """Set a patch property with correct type, size, etc."""

        self._check_pid(bid, pid)

        _check_pp_name(name)

        # Determine correct size
        patch = self.get_patch(bid, pid)
        dk = patch.ken - patch.kst
        dj = patch.jen - patch.jst
        di = patch.ien - patch.ist

        # Preallocate output in correct squeezed k j i form
        val_out = np.squeeze(np.ones((dk, dj, di))).astype(np.float32)

        if _is_scalar(val):

            val_out[:] = val

        else:

            # Check size of val
            shape_out = (di, dj, dk)
            shape_in = np.shape(val)
            if shape_out == shape_in:
                val_out[:] = np.squeeze(np.swapaxes(val, 0, 2))
            else:
                raise Exception(
                    "Bad patch property input size %s, should be %s"
                    % (str(shape_in), str(shape_out))
                )

        TstreamGrid.set_pp(self, name, TYPE.float, bid, pid, val_out)

    def set_pv(self, name, bid, pid, val):
        """Set a patch variable with correct type."""

        self._check_pid(bid, pid)

        _check_pv_name(name)

        try:
            var_type = PV_TYPES[name]
            if var_type == TYPE.int:
                var_val = _convert_to_int(val)
            else:
                var_val = _convert_to_float(val)

            TstreamGrid.set_pv(self, name, var_type, bid, pid, var_val)

        except KeyError:
            raise Exception("Invalid patch variable: %s" % name)

    def set_bp(self, name, bid, val):
        """Set block property ensuring correct types."""

        self._check_bid(bid)

        _check_bp_name(name)

        # Check size of input coordinates
        b = self.get_block(bid)
        ni, nj, nk = (b.ni, b.nj, b.nk)
        shape_out = (ni, nj, nk)
        shape_in = np.shape(val)
        if shape_out == shape_in:
            val_out = np.ones((nk, nj, ni), dtype=np.float32)
            val_out[:] = np.swapaxes(val, 0, 2)
        else:
            raise Exception(
                "Bad block property input size %s, should be %s"
                % (str(shape_in), str(shape_out))
            )

        TstreamGrid.set_bp(self, name, TYPE.float, bid, val_out)

    def get_pv(self, name, bid, pid):
        """Fetch patch variable with error checking."""

        _check_pv_name(name)

        self._check_pid(bid, pid)
        pv = TstreamGrid.get_pv(self, name, bid, pid)

        return pv

    def get_bp(self, name, bid, raise_missing=True):
        """Retrieve block property by name/ bid, default to NaN if not present."""

        self._check_bid(bid)

        _check_bp_name(name)

        try:
            bp = TstreamGrid.get_bp(self, name, bid)
        except KeyError:
            if raise_missing:
                raise Exception(
                    "Block property not found: %s on bid %d"
                    % (name, bid)
                )
            else:
                bp = (
                    np.ones_like(TstreamGrid.get_bp(self, "x", bid))
                    * np.nan
                )

        return np.swapaxes(bp,0,2)

    def get_bv(self, name, bid):
        """Retrieve block property by name/ bid, default to NaN if not present."""

        self._check_bid(bid)

        _check_bv_name(name)

        try:
            bp = TstreamGrid.get_bv(self, name, bid)
        except KeyError:
            raise Exception(
                "Block variable not found: %s on bid %d" % (name, bid)
            )

        return bp

    #
    # Patching
    #

    def cut_patch(self, bid, pid, squeeze=True, offset=0):
        """Structured cut a patch from grid."""
        P = self.get_patch(bid, pid)

        ist = P.ist + 0
        ien = P.ien + 0
        jst = P.jst + 0
        jen = P.jen + 0
        kst = P.kst + 0
        ken = P.ken + 0

        # Perform an offset if requested
        if offset:

            dk = ken - kst
            dj = jen - jst
            di = ien - ist

            if di == 1:
                if not ist == 0:
                    offset *= -1
                ist += offset
                ien += offset

            elif dj == 1:
                if not jst == 0:
                    offset *= -1
                jst += offset
                jen += offset

            elif dk == 1:
                if not kst == 0:
                    offset *= -1
                kst += offset
                ken += offset

            else:
                raise Exception(
                    "Cannot offset patch cut, no thin dimension!"
                )

        return Cut(self, bid, ist, ien, jst, jen, kst, ken, squeeze)

    def find_patches(self, kind=None):
        """Return block and patch ids of all patches of a given kind."""
        bid = []
        pid = []
        for b in self.bids:
            for p in self.pids(b):
                if (kind is None) or (
                    self.get_patch(b, p).kind == kind
                ):
                    bid.append(b)
                    pid.append(p)

        return bid, pid

    def make_patch(
        self, kind, bid, i, j, k, nxbid=0, nxpid=0, dirs=None
    ):
        """Create a patch of a specified kind on a given block."""

        p = TYPE.TstreamPatch()

        p.kind = getattr(KIND, kind)

        p.bid = bid

        p.ist, p.ien = i
        p.jst, p.jen = j
        p.kst, p.ken = k

        p.nxbid = nxbid
        p.nxpid = nxpid

        if dirs is not None:
            p.idir, p.jdir, p.kdir = dirs
        else:
            p.idir, p.jdir, p.kdir = (0, 1, 2)

        p.nface = 0
        p.nt = 1

        new_pid = TstreamGrid.add_patch(self, bid, p)
        p.pid = new_pid

        return p, new_pid

    #
    # Initial guess
    #

    def _guess_block(self, bid, x, Po, To, Ma, Al):
        """Apply initial guess to a single block."""

        # Gas props
        ga = self.get_av("ga")
        cp = self.get_av("cp")
        rgas = cp * (ga - 1.0) / ga
        cv = cp / ga

        # Coordinates
        xb = self.get_bp("x", bid)
        rb = self.get_bp("r", bid)

        # Interpolate guess to block coords
        Pob = np.interp(xb, x, Po)
        Tob = np.interp(xb, x, To)
        Mab = np.interp(xb, x, Ma)
        Alb = np.interp(xb, x, Al)

        # Get velocities
        Vb = compflow.V_cpTo_from_Ma(Mab, ga) * np.sqrt(cp * Tob)
        Vxb = Vb * np.cos(np.radians(Alb))
        Vtb = Vb * np.sin(np.radians(Alb))
        Vrb = np.zeros_like(Vb)

        # Static pressure and temperature
        Pb = Pob / compflow.Po_P_from_Ma(Mab, ga)
        Tb = Tob / compflow.To_T_from_Ma(Mab, ga)

        # Density
        rob = Pb / rgas / Tb

        # Energy
        eb = cv * Tb + (Vb ** 2.0) / 2

        # Primary vars
        rovxb = rob * Vxb
        rovrb = rob * Vrb
        rorvtb = rob * rb * Vtb
        roeb = rob * eb

        # Apply to grid
        self.set_bp("ro", bid, rob)
        self.set_bp("rovx", bid, rovxb)
        self.set_bp("rovr", bid, rovrb)
        self.set_bp("rorvt", bid, rorvtb)
        self.set_bp("roe", bid, roeb)

    def guess_1d(self, x, Po, To, Ma, Al):
        """Apply initial guess to all blocks."""
        for bid in self.bids:
            self._guess_block(bid, x, Po, To, Ma, Al)
        self.set_av("restart", 1)

    def guess_file(self, fname):
        """Set initial guess from a file."""

        if not os.path.exists(fname):
            raise Exception("Guess file does not exist %s" % fname)
        gg = read_hdf5(fname)

        # Check for same number of blocks
        if not gg.nblock == self.nblock:
            raise Exception(
                "Guess file has wrong number of blocks: %d, should be %d"
                % (gg.nblock, self.nblock)
            )

        # Use existing Poisson and flow fields
        self.set_av("poisson_nstep", 0)
        self.set_av("poisson_restart", 1)
        self.set_av("restart", 1)

        # Copy varaibles
        for var in [
            "ro",
            "rovx",
            "rovr",
            "rorvt",
            "roe",
            "trans_dyn_vis",
            "phi",
        ]:
            for bid, nijk, nijkg in zip(self.bids, self.nijk, gg.nijk):

                valg = gg.get_bp(var, bid)

                if not nijk == nijkg:
                    # If block sizes do not match, then interpolate by index

                    # Guess block relative indexes
                    ijkgv = [np.linspace(0.0, 1.0, n) for n in nijkg]

                    # Target block relative indexes
                    ijkv = [np.linspace(0.0, 1.0, n) for n in nijk]
                    ijk = np.stack(
                        np.meshgrid(*ijkv, indexing="ij"), axis=-1
                    )

                    val = interpn(ijkgv, valg, ijk).astype(np.float32)

                else:
                    val = valg

                self.set_bp(var, bid, val)

        # Throw out spurious negative wall distances
        for bid in self.bids:
            phi = self.get_bp("phi", bid)
            if np.any(phi[:] < 0.0):
                phi[phi < 0.0] = 0.0
                self.set_bp("phi", bid, phi)

    #
    # Boundary conditions
    #

    def apply_inlet_uniform(self, Po, To, pitch, yaw):
        """Uniform inlet boundary condition."""
        spf = np.array([0.0, 1.0])
        self.apply_inlet_1d(spf, Po, To, pitch, yaw)

    def apply_inlet_1d(self, spf, Po, To, pitch, yaw):
        """Interpolate boundary conditions as span fraction on all inlet blocks."""

        # Replicate scalar inputs
        if _is_scalar(Po):
            Po = np.ones_like(spf) * Po
        if _is_scalar(To):
            To = np.ones_like(spf) * To
        if _is_scalar(pitch):
            pitch = np.ones_like(spf) * pitch
        if _is_scalar(yaw):
            yaw = np.ones_like(spf) * yaw

        # Make interpolators
        func_Po = interp1d(spf, Po)
        func_To = interp1d(spf, To)
        func_pitch = interp1d(spf, pitch)
        func_yaw = interp1d(spf, yaw)

        # Loop over all patches and do interpolation
        for bid, pid in zip(*self.find_patches(KIND.inlet)):

            r = self.get_bp("r", bid)

            coords_patch = self.cut_patch(bid, pid, squeeze=False)
            r_patch = coords_patch.r
            spf_patch = (r_patch - r.min()) / (r.max() - r.min())

            self.set_pp("pstag", bid, pid, func_Po(spf_patch))
            self.set_pp("tstag", bid, pid, func_To(spf_patch))
            self.set_pp("pitch", bid, pid, func_pitch(spf_patch))
            self.set_pp("yaw", bid, pid, func_yaw(spf_patch))

            self.set_pv("rfin", bid, pid, 0.5)
            self.set_pv("sfinlet", bid, pid, 0.1)

    def apply_outlet(self, Pout, ipout=3):
        """Apply outlet static pressure to all outlet patches."""
        for bid, pid in zip(*self.find_patches(KIND.outlet)):
            self.set_pv("throttle_type", bid, pid, 0)
            self.set_pv("ipout", bid, pid, ipout)
            self.set_pv("pout", bid, pid, Pout)

    def apply_rotation(self, row_types, rpms=None):
        """Set wall rotations."""
        for i in range(self.nrow):

            bids = self.row_bids[i]

            if rpms:
                rpm = rpms[i]
            else:
                rpm = self.get_bv("rpm", bids[0])

            if row_types[i] == "stationary":
                self.set_bv("rpmi1", 0.0, bids)
                self.set_bv("rpmi2", 0.0, bids)
                self.set_bv("rpmj1", 0.0, bids)
                self.set_bv("rpmj2", 0.0, bids)
                self.set_bv("rpmk1", 0.0, bids)
                self.set_bv("rpmk2", 0.0, bids)
            elif row_types[i] == "tip_gap":
                self.set_bv("rpmi1", rpm, bids)
                self.set_bv("rpmi2", rpm, bids)
                self.set_bv("rpmj1", rpm, bids)
                self.set_bv("rpmj2", 0.0, bids)
                self.set_bv("rpmk1", rpm, bids)
                self.set_bv("rpmk2", rpm, bids)
            elif row_types[i] == "shroud":
                self.set_bv("rpmi1", rpm, bids)
                self.set_bv("rpmi2", rpm, bids)
                self.set_bv("rpmj1", rpm, bids)
                self.set_bv("rpmj2", rpm, bids)
                self.set_bv("rpmk1", rpm, bids)
                self.set_bv("rpmk2", rpm, bids)
            else:
                raise Exception("Unknown row type %s", row_types[i])

    def apply_nblade(self, nbs):
        """Set numbers of blades by row."""
        for bids, nb in zip(self.row_bids, nbs):
            self.set_bv("fblade", nb, bids)
            self.set_bv("nblade", nb, bids)

    def apply_xllim(self, fracs):
        """Set mixing length limits by row."""
        for bids, frac in zip(self.row_bids, fracs):

            nb = self.get_bv("nblade", bids[0])
            r = self.get_bp("r", bids[0])
            rm = np.mean([r.max(), r.min()])
            pitch = 2.0 * np.pi * rm / float(nb)
            self.set_bv("xllim", pitch * frac, bids)

    def make_block(self, x, r, rt):
        """Make a new block from coordinates, add to the grid."""

        bid = (
            self.get_nb()
        )  # Next block id is equal to current num of blocks
        b = TYPE.TstreamBlock()
        b.bid = bid
        b.ni, b.nj, b.nk = np.shape(x)
        b.np = 0
        b.procid = 0
        b.threadid = 0

        # Add to grid and set coordinates as block properties
        self.add_block(b)
        for vname, vval in zip(["x", "r", "rt"], [x, r, rt]):
            self.set_bp(vname, bid, vval)

        return b, bid

    #
    # Misc
    #

    def load_balance(self, ngpu, comm_weight=1.0):
        """Load balance the grid over GPUs."""
        with suppress_print():
            ts_tstream_load_balance.load_balance(
                self, ngpu, comm_weight
            )

    def write_hdf5(self, fname):
        """Write out an hdf5 file."""

        # Turbostream prefers lager drinkers
        self.set_av("if_ale", 0)

        with suppress_print():
            TstreamGrid.write_hdf5(self, fname)


def set_default_variables(g):
    # Load official defaults
    for name, val in ts_tstream_default.av.items():
        g.set_av(name, val)
    for name, val in ts_tstream_default.bv.items():
        g.set_bv(name, val)


class OHGrid(Grid):
    """Turbostream grid with O-H topology meshed in Autogrid."""

    @classmethod
    def convert(cls, g):
        """Mutate a general grid object into an O-H topology grid."""
        g.__class__ = cls

    @property
    def omesh_bids(self):
        oblocks = []
        # Exclude tip block by looking for nomatch patch
        bpid_per = self.find_patches(KIND.periodic)
        for bid, pid in zip(*bpid_per):
            P = self.get_patch(bid, pid)
            if (
                P.nxbid == bid
                and bid not in oblocks
                and not (bid in self.tip_gap_bids)
            ):
                CP1 = self.cut_patch(bid, pid)
                CP2 = self.cut_patch(P.nxbid, P.nxpid)
                if np.isclose(np.mean(CP1.rt), np.mean(CP2.rt)):
                    oblocks.append(bid)
        return oblocks

    @property
    def trailing_edge_bids(self):
        # Choose TE bids by looking for two patches to o block
        te_bids = []

        # Loop over bids
        for bid in self.bids:
            n = 0
            if bid in self.omesh_bids:
                continue
            # Count the number of patches on bid that connect to an o block
            for pid in self.pids(bid):
                P = self.get_patch(bid, pid)
                if P.nxbid in self.omesh_bids:
                    n += 1
            if n == 2:
                te_bids.append(bid)
        return te_bids

    @property
    def tip_gap_bids(self):
        return self.find_patches(KIND.nomatch)[0]

    def cut_blade_surfs(self, normalise=False):
        cuts = [
            Cut(self, bid, 0, -1, 0, -1, 0, 1)
            for bid in self.omesh_bids
        ]
        for ci in cuts:
            ci.zeta = _calculate_zeta(ci.x, ci.rt, normalise)
        return cuts

    def cut_span(self, spf):
        j_cut = self.span_fraction_index(spf)

        # # # Check if our desired j-index is inside the tip gap
        # # nj = self.nj[0]
        # # bid_tip = self.tip_gap_bids
        # # nj_tip = [self.nj[b] for b in bid_tip]
        # # print(nj_tip)
        # # rsrtr
        # if j_cut > (nj - nj_tip):
        #     bid_cut = self.bids
        #     j_all = np.ones_like(bid_cut,dtype=int)*j_cut
        #     j_all[np.isin(bid_cut, bid_tip)] = j_cut - (nj - nj_tip)
        # else:
        #     bid_cut = np.setdiff1d(self.bids, bid_tip)
        #     j_all = np.ones_like(bid_cut,dtype=int)*j_cut

        return [Cut(self, bid, ist=0, ien=-1, jst=j_cut, jen=j_cut+1, kst=0, ken=-1) for bid in self.bids]


class HGrid(Grid):
    """Turbostream grid with H topology manually meshed."""

    @classmethod
    def convert(cls, g):
        """Mutate a general grid object into an H topology grid."""
        g.__class__ = cls

    @property
    def _ilte(self):
        ile = []
        ite = []
        for bid in self.bids:
            C1 = Cut(self, bid, 0, -1, 0, 1, 0, 1)
            C2 = Cut(self, bid, 0, -1, 0, 1, -2, -1)
            pitch = 2.0 * np.pi / float(self.get_bv("nblade", bid))
            dt = np.squeeze(C1.t - C2.t + pitch)
            tol = 1e-6
            ile.append(np.argmax(dt > tol) - 1)
            ite.append(len(dt) - np.argmax(np.flip(dt) > tol) - 1)
        return ile, ite

    @property
    def ile(self):
        """Indices to leading edges of all blade rows."""
        return self._ilte[0]

    @property
    def ite(self):
        """Indices to trailing edges of all blade rows."""
        return self._ilte[1]

    def cut_blade_sides(self):
        """Nested list of pressure/suction side cuts in each row."""
        cuts = []
        for bid in self.bids:
            C1 = Cut(
                self, bid, self.ile[bid], self.ite[bid], 0, -1, 0, 1
            )
            C2 = Cut(
                self, bid, self.ile[bid], self.ite[bid], 0, -1, -2, -1
            )
            C12 = [C1, C2]
            for ci in C12:
                ci.zeta = _calculate_zeta(ci.x, ci.rt)
            cuts.append(C12)
        return cuts

    def cut_blade_surfs(self):
        cuts = self.cut_blade_sides()
        cut_out = []
        for cut in cuts:

            pitch = 2.0 * np.pi / float(cut[0].nblade)
            iu = np.argmax([ci.t.max() for ci in cut])
            cut[iu].t -= pitch
            cut[iu].rt = cut[iu].r * cut[iu].t
            cut_now = _join_cuts(cut, axis=0, flip=1)

            cut_now.zeta = _calculate_zeta(cut_now.x, cut_now.rt)

            cut_out.append(cut_now)

        return cut_out
