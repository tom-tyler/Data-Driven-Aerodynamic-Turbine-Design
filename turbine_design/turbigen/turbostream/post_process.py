"""Post process a steady Turbostream solution"""
import numpy as np
import turbigen.compflow_native as compflow
import sys, os, json
from ts import ts_tstream_cut
from turbigen import average
from . import grid, cooling
import matplotlib.pyplot as plt

Pref = 1e5
Tref = 300.0

# Choose which variables to write out
varnames = ["x", "rt", "eff_lost", "pfluc"]


class suppress_print:
    """A context manager that temporarily sets STDOUT to /dev/null."""

    def __enter__(self):
        self.orig_out = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stdout = self.orig_out


def _get_AR(C):
    xr_hub = np.stack([v[:, 0].reshape(-1) for v in [C.x, C.r]])
    xr_cas = np.stack([v[:, -1].reshape(-1) for v in [C.x, C.r]])
    xr_hub = xr_hub[:, xr_hub[0].argsort()]
    xr_cas = xr_cas[:, xr_cas[0].argsort()]
    x = np.linspace(C.x.min(), C.x.max())
    r_hub = np.interp(x, *xr_hub)
    r_cas = np.interp(x, *xr_cas)
    return float(np.trapz(r_cas - r_hub, x) / x.ptp() ** 2.0)


def node_to_face(cut, prop_name):
    """For a (n,m) matrix of some property, average over the four corners of
    each face to produce an (n-1,m-1) matrix of face-centered properties."""
    return np.mean(
        np.stack(
            (
                getattr(cut, prop_name)[:-1, :-1].astype(float),
                getattr(cut, prop_name)[1:, 1:].astype(float),
                getattr(cut, prop_name)[:-1, 1:].astype(float),
                getattr(cut, prop_name)[1:, :-1].astype(float),
            )
        ),
        axis=0,
    )


def face_length_vec(c):
    """For a matrix of coordinates, get face length matrices along each dim."""
    return c[1:, 1:] - c[:-1, :-1], c[:-1, 1:] - c[1:, :-1]


def face_area(cut):
    """Calculate x and r areas for all cells in a cut."""
    (dx1, dx2), (dr1, dr2), (drt1, drt2) = [
        face_length_vec(c) for c in (cut.x, cut.r, cut.rt)
    ]

    Ax = 0.5 * (dr1 * drt2 - dr2 * drt1)
    Ar = 0.5 * (dx2 * drt1 - dx1 * drt2)

    return Ax, Ar


def mix_out(cut):
    """Take a structured cuts and mix out the flow at constant A."""

    # Gas properties
    cp = cut.cp
    ga = cut.ga
    rgas = cp * (ga - 1.0) / ga
    Omega = cut.rpm / 60.0 * 2.0 * np.pi

    # Do the mixing
    r_mix, ro_mix, rovx_mix, rovr_mix, rorvt_mix, roe_mix, Ax = average.mix_out(
        cut.x,
        cut.r,
        cut.rt,
        cut.ro,
        cut.rovx,
        cut.rovr,
        cut.rorvt,
        cut.roe,
        ga,
        rgas,
        Omega,
    )

    # Secondary mixed vars
    vx_mix, vr_mix, vt_mix, P_mix, T_mix = average.primary_to_secondary(
        r_mix, ro_mix, rovx_mix, rovr_mix, rorvt_mix, roe_mix, ga, rgas
    )

    # Max a new cut with the mixed out flow
    cut_out = ts_tstream_cut.TstreamStructuredCut()

    cut_out.pref = cut.pref
    cut_out.tref = cut.tref
    cut_out.ni = 1
    cut_out.nj = 1
    cut_out.nk = 1
    cut_out.ga = ga
    cut_out.cp = cp
    cut_out.ifgas = 0
    cut_out.write_egen = 0
    cut_out.nblade = cut.nblade

    cut_out.rpm = cut.rpm
    cut_out.x = np.mean(cut.x)
    cut_out.r = r_mix
    cut_out.rt = np.mean(cut.rt)
    cut_out.ro = ro_mix
    cut_out.rovx = rovx_mix
    cut_out.rovr = rovr_mix
    cut_out.rorvt = rorvt_mix
    cut_out.roe = roe_mix

    cut_out.tstat = T_mix
    cut_out.pstat = P_mix

    cut_out.vx = vx_mix
    cut_out.vr = vr_mix
    cut_out.vt = vt_mix
    cut_out.vabs = np.sqrt(vx_mix ** 2.0 + vr_mix ** 2.0 + vt_mix ** 2.0)
    cut_out.U = r_mix * Omega
    cut_out.vt_rel = vt_mix - cut_out.U

    cut_out.vabs_rel = np.sqrt(
        cut_out.vx ** 2.0 + cut_out.vr ** 2.0 + cut_out.vt_rel ** 2.0
    )
    cut_out.mach_rel = cut_out.vabs_rel / np.sqrt(ga * rgas * cut_out.tstat)

    cut_out.mach = cut_out.vabs / np.sqrt(ga * rgas * cut_out.tstat)
    cut_out.pstag = compflow.Po_P_from_Ma(cut_out.mach, ga) * cut_out.pstat
    cut_out.pstag_rel = compflow.Po_P_from_Ma(cut_out.mach_rel, ga) * cut_out.pstat
    cut_out.tstag = compflow.To_T_from_Ma(cut_out.mach, ga) * cut_out.tstat

    cut_out.yaw = np.degrees(np.arctan2(cut_out.vt, cut_out.vx))
    cut_out.yaw_rel = np.degrees(np.arctan2(cut_out.vt_rel, cut_out.vx))

    cut_out.entropy = cp * np.log(cut_out.tstat / cut_out.tref) - rgas * np.log(
        cut_out.pstat / cut_out.pref
    )

    cut_out.Ax = Ax
    cut_out.mdot = rovx_mix * Ax * cut_out.nblade

    return cut_out


def cut_by_indices(g, bid, ijk_sten):
    """Structured cut from grid, allowing for end indices with -1.

    Parameters
    ----------
    g :
        The Turbostream grid object.
    bid : int
        Block id in which to take cut.
    ijk_sten: (3, 2) array, int
        First col ijk start indices, second col ijk end indices.

    """

    # Assemble end indices nijk for desired block
    blk = g.get_block(bid)
    nijk = np.tile([blk.ni, blk.nj, blk.nk], (2, 1)).T

    # Correct the end indices
    ijk_sten = np.array(ijk_sten)
    ijk_sten[ijk_sten < 0] = nijk[ijk_sten < 0] + ijk_sten[ijk_sten < 0]

    ijk_sten[:, 1] += 1

    cut = ts_tstream_cut.TstreamStructuredCut()
    cut.read_from_grid(g, Pref, Tref, bid, *ijk_sten.flat)

    return cut


def cut_by_patch(g, bid, pid):
    """Structured cut a patch from grid."""

    P = g.get_patch(bid, pid)
    cut = ts_tstream_cut.TstreamStructuredCut()
    cut.read_from_grid(g, Pref, Tref, bid, P.ist, P.ien, P.jst, P.jen, P.kst, P.ken)

    return cut


def cut_rows_mixed(g):
    """Mixed-out cuts at row inlet and exit"""

    cuts = sum(g.cut_rows(), [])
    return [mix_out(c) for c in cuts]


def _integrate_length(chi):
    """Integrate quadratic camber line length given angles."""
    xhat = np.linspace(0.0, 1.0)
    tanchi_lim = np.tan(np.radians(chi))
    tanchi = np.diff(tanchi_lim) * xhat + tanchi_lim[0]
    return np.trapz(np.sqrt(1.0 + tanchi ** 2.0), xhat)


def find_chord(g, bid):
    """Determine axial chord of a row."""

    x, r, rt = [
        np.swapaxes(g.get_bp(vi, bid), 0, -1)[:, 2, (0, -1)] for vi in ["x", "r", "rt"]
    ]
    dt = np.diff(rt / r, 1, axis=1).flat
    pitch = dt[0]
    is_blade = dt / pitch < 0.995
    ile = np.argmax(is_blade) - 1
    is_blade_2 = dt / pitch < 0.995
    is_blade_2[: (ile + 1)] = True
    ite = np.argmax(~is_blade_2)
    cx = x[ile:ite, 0].ptp()

    return cx, ile, ite, pitch


def extract_surf(g, bid):
    cx, ile, ite, _ = find_chord(g, bid)
    C = cut_by_indices(g, bid, [[ile, ite], [2, 2], [0, -1]])
    P = np.moveaxis(C.pstat, 0, -1)[:, (0, -1)].astype(float)
    x = np.moveaxis(C.x, 0, -1)[:, (0, -1)].astype(float)
    rt = np.moveaxis(C.rt, 0, -1)[:, (0, -1)]
    surf = np.cumsum(
        np.sqrt(np.diff(x, 1, 0) ** 2.0 + np.diff(rt, 1, 0) ** 2.0), axis=0
    )
    surf = np.insert(surf, 0, np.zeros((1, 2)), axis=0).astype(float)
    return np.squeeze(surf), np.squeeze(P), np.squeeze(x)


def circ_coeff(C, Po1, P2):

    P = C.pstat
    zeta = C.zeta
    r = np.mean(C.r, axis=0)

    Cp = (Po1 - P) / (Po1 - P2)
    Cp[Cp < 0.0] = 0.0
    sqrt_Cp = np.sqrt(Cp)

    upper = np.trapz(np.where(zeta < 0.0, sqrt_Cp, 0.0), zeta, axis=0)
    lower = np.trapz(np.where(zeta >= 0.0, sqrt_Cp, 0.0), zeta, axis=0)

    So_r = np.max(np.abs(zeta), axis=0)
    Co_r = np.abs(upper - lower) / So_r

    So = np.asscalar(np.trapz(So_r, r) / r.ptp())
    Co = np.asscalar(np.trapz(Co_r, r) / r.ptp())

    return Co, So


def post_process(output_hdf5):
    """Do the post processing on a given hdf5"""

    basedir = os.path.dirname(os.path.abspath(output_hdf5))
    run_name = os.path.split(os.path.abspath(basedir))[-1]

    if not os.path.exists(output_hdf5):
        inst_hdf5 = output_hdf5.replace("_avg", "")
        if not os.path.exists(inst_hdf5):
            raise Exception("No output hdf5 found.")
        else:
            output_hdf5 = inst_hdf5

    # Load the flow solution
    g = grid.read_hdf5(output_hdf5)

    nan_str = os.popen("grep NAN %s/log.txt" % basedir).read()
    if np.any(np.isnan(g.get_bp("ro", 0))) or ("NAN" in nan_str):

        print("Simulation NaN'd, exiting.")
        sys.exit(37)

    # Gas properties
    cp = g.get_av("cp")  # Specific heat capacity at const p
    ga = g.get_av("ga")  # Specific heat ratio
    gm1_g = (ga-1.)/ga

    # 1D mixed-out average cuts for stator/rotor inlet/outlet
    cut_all = cut_rows_mixed(g)
    sta_in, sta_out, rot_in, rot_out = cut_all

    # Blade speed
    rpm = np.max([g.get_bv("rpm", b) for b in g.get_block_ids()])
    Omega = rpm / 60.0 * 2.0 * np.pi
    U = Omega * rot_in.r

    # Pull out useful flow vars
    c13 = [sta_in, rot_out]
    To1, To3 = [c.tstag for c in c13]
    Po1, Po3 = [c.pstag for c in c13]
    ho1, ho3 = [c.tstag * cp for c in c13]

    # Enthalpy flows
    Hoc, mdotc = cooling.sum_coolant_enthalpy(g)
    mdot1 = sta_in.rovx * sta_in.Ax * sta_in.nblade
    mdot3 = mdotc + mdot1  # Make sure mass is conserved
    Ho1 = mdot1 * ho1
    Ho3 = mdot3 * ho3

    # Calculate stage loading coefficient based on power per inlet mass flow
    Power = Ho1+Hoc-Ho3
    Psi = Power / mdot1 / U ** 2.0

    # Polytropic efficiency
    eff_poly = (
        ga
        / (ga - 1.0)
        * np.log(rot_out.tstag / sta_in.tstag)
        / np.log(rot_out.pstag / sta_in.pstag)
    )

    # Isentropic efficiency, no cooling
    To3s = To1*(Po3/Po1)**gm1_g
    eff_isen = (To3 / To1 - 1.0) / (To3s/To1 - 1.0)

    # Young and Horlock (2006) cooled turbine efficiency
    # Mix the main-stream and coolant flows, First Law
    Tom = (Ho1 + Hoc)/mdot3/cp
    # Now the weighted-pressure definition of mixture pressure
    Pom = cooling.weighted_pressure(g, mdot1, Po1)
    # Expand isentropically to pos
    To3sw = Tom * (Po3/Pom)**gm1_g
    eff_wp = Power / mdot3 / cp / (Tom - To3sw)

    # Reaction
    Lam = (rot_out.tstat - rot_in.tstat) / (rot_out.tstat - sta_in.tstat)

    # Flow angles
    Al = [ci.yaw for ci in cut_all]
    Al_rel = [ci.yaw_rel for ci in cut_all]

    # Viscosity
    if g.get_av("viscosity_law"):
        muref = g.get_av("viscosity")
        Tref = 288.0
        expon = 0.62
        T2 = sta_out.tstat
        mu2 = muref * (T2 / Tref) ** expon
    else:
        mu2 = g.get_av("viscosity")

    # Reynolds num
    ro2 = sta_out.ro
    V2 = sta_out.vabs
    cx = np.array(g.cx)
    Re_cx = ro2 * V2 * cx / mu2
    ell_cx = _integrate_length(Al[:2])
    Re_ell = (Re_cx * ell_cx).tolist()

    # Circulation coefficient
    Pres = ((sta_in.pstag, sta_out.pstat), (rot_in.pstag_rel, rot_out.pstat))
    Co, So = zip(*[circ_coeff(C, *PoP) for C, PoP in zip(g.cut_blade_surfs(), Pres)])
    Co = list(Co)
    So = list(So)

    Re_So = (Re_cx * So[0] / cx).tolist()

    # Pitch to chord
    bids = g.get_block_ids()[0], g.get_block_ids()[-1]
    nb = np.array([g.get_bv("nblade", bid) for bid in bids])
    pitch_rt = np.array(
        [2.0 * np.pi * Ci.r / nbi for Ci, nbi in zip([sta_in, rot_in], nb)]
    )
    s_cx = (pitch_rt / cx).tolist()

    # Loss coefficients
    Ypv = (sta_in.pstag - sta_out.pstag) / (sta_in.pstag - sta_out.pstat)
    Ypb = (rot_in.pstag_rel - rot_out.pstag_rel) / (rot_in.pstag_rel - rot_out.pstat)
    Yp = [Ypv, Ypb]

    # Loss ratio
    loss_rat = (rot_in.entropy - sta_in.entropy) / (rot_out.entropy - sta_in.entropy)

    # Axial velocity ratio
    zeta = (sta_in.vx / sta_out.vx, rot_out.vx / sta_out.vx)

    # Convergence residual
    resid_str = os.popen(
        "grep 'TOTAL DAVG' %s/log.txt | tail -10 | cut -d ' ' -f3" % basedir
    ).read()
    resid = np.array([float(ri) for ri in resid_str.splitlines()]).mean()

    err_Cp_fit = plot_Cp(g, basedir)

    # Aspect ratio
    AR = [_get_AR(C) for C in g.cut_blade_surfs()]

    # Save metadata in dict
    meta = {
        "Al": Al,
        "Alrel": Al_rel,
        "psi": Psi,
        "eta": eff_poly,
        "eta_lost": 1.0 - eff_poly,
        "eta_isen": eff_isen,
        "eta_wp": eff_wp,
        "eta_lost_wp": 1.0-eff_wp,
        "runid": run_name,
        "Ma2": sta_out.mach,
        "phi": sta_out.vx / U,
        "Lam": Lam,
        "Re": Re_ell,
        "Re_cx": Re_cx.tolist(),
        "Re_So": Re_So,
        "Co": Co,
        "resid": resid,
        "s_cx": s_cx,
        "Yp": Yp,
        "zeta": zeta,
        "loss_rat": loss_rat,
        "err_Cp_fit": err_Cp_fit,
        "AR": AR,
    }

    with open(os.path.join(basedir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=4)

    # Continue to make graphs if the argument is specified
    if not "--plot" in sys.argv:
        return

    # Lazy import
    import matplotlib.pyplot as plt

    # Pressure distributions
    # Mach number contours
    eta_lost_pc = (1.0 - eff_poly) * 100.0
    fig, ax = plt.subplots()
    title_string = (
        ", ".join(["%s=%.2f" % (v, meta[v]) for v in ["phi", "psi", "Lam", "Ma2"]])
        + ", Co=%.2f,%.2f" % tuple(meta["Co"])
        + ", lost eta = %.1f\%%" % eta_lost_pc
    )
    ax.set_title(title_string, fontsize=12)

    _, Pv, xv = extract_surf(g, 0)
    Po1 = sta_in.pstag
    P2 = sta_out.pstat
    Cpv = (Pv - Po1) / (Po1 - P2)

    _, Pb, xb = extract_surf(g, 1)
    Po2 = rot_in.pstag_rel
    P3 = rot_out.pstat
    Cpb = (Pb - Po2) / (Po2 - P3)

    ax.plot(xv, Cpv)
    ax.plot(xb, Cpb)

    ax.set_xlabel("Axial Chord")
    ax.set_ylabel("Static Pressure Coefficient")

    plt.tight_layout()
    plt.savefig(os.path.join(basedir, "Cp.pdf"))


def get_Cp(g, spf, irow):
    crow = cut_rows_mixed(g)
    c = g.cut_blade_surfs()[irow]

    Poref = [crow[0].pstag, crow[2].pstag_rel][irow]
    Pref = [crow[1].pstat, crow[3].pstat][irow]
    j = g.span_fraction_index(spf)

    P = c.pstat[:, j]
    Cp = (P - Poref) / (Poref - Pref)

    z = c.zeta[:, j]
    istag = np.argmax(P)
    z = z - z[istag]

    if irow:
        Cpss = Cp[z >= 0.0]
        zss = z[z >= 0.0]
        Cpps = np.flip(Cp[z <= 0.0])
        zps = np.flip(-z[z <= 0.0])
    else:
        Cpss = np.flip(Cp[z <= 0.0])
        zss = np.flip(-z[z <= 0.0])
        Cpps = Cp[z >= 0.0]
        zps = z[z >= 0.0]

    zps = zps / zps[-1]
    zss = zss / zss[-1]

    return zss, Cpss, zps, Cpps


def plot_Cp(g, base_dir):
    """Write out surface pressure distributions."""

    strs = ["stator", "rotor"]

    for irow in range(2):

        zss, Cpss, zps, Cpps = get_Cp(g, 0.5, irow)

        fig, ax = plt.subplots()
        ax.plot(zss, Cpss)
        ax.plot(zps, Cpps)
        ax.set_xlabel("Surface Length Fraction")
        ax.set_ylabel("Static Pressure Coefficient")
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, "Cp_zeta_%s.pdf" % strs[irow]))
