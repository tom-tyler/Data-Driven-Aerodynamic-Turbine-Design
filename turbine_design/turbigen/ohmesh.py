"""Produce g and bcs files from geometry and mesh config files."""

import os, shutil, subprocess
import numpy as np
from . import hmesh, geometry
from time import sleep

REMOTE = "gp-111"
SSH_ENV_VARS = ["SSH_AUTH_SOCK", "SSH_AGENT_PID"]


def write_geomturbo(fname, ps, ss, h, c, nb, tips, cascade=False):
    """Write blade and annulus geometry to AutoGrid GeomTurbo file.

    Parameters
    ----------

    fname : File name to write
    ps    : Nested list of arrays of pressure-side coordinates,
            ps[row][section][point on section, x/r/rt]
            We allow different sizes for each section and row.
    ss    : Same for suction-side coordinates.
    h     : Array of hub line coordinates, h[axial location, x/r].
    c     : Same for casing line.
    nb    : Iterable of numbers of blades for each row."""

    # Determine numbers of points
    ni_h = np.shape(h)[0]
    ni_c = np.shape(c)[0]

    n_row = len(ps)
    n_sect = [len(psi) for psi in ps]
    ni_ps = [[np.shape(psii)[0] for psii in psi] for psi in ps]
    ni_ss = [[np.shape(ssii)[0] for ssii in ssi] for ssi in ss]

    fid = open(fname, "w")

    # # Autogrid requires R,X,T coords
    # ps = ps[i][:,[1,0,2]]
    # ss = ss[i][:,[1,0,2]]

    if cascade:
        # Swap the coordinates
        for i in range(n_row):
            for k in range(n_sect[i]):
                ps[i][k] = ps[i][k][:, (1, 2, 0)]
                ss[i][k] = ss[i][k][:, (1, 2, 0)]
    else:
        # Convert RT to T
        for i in range(n_row):
            for k in range(n_sect[i]):
                ps[i][k][:, 2] = ps[i][k][:, 2] / ps[i][k][:, 1]
                ss[i][k][:, 2] = ss[i][k][:, 2] / ss[i][k][:, 1]

    # Write the header
    fid.write("%s\n" % "GEOMETRY TURBO")
    fid.write("%s\n" % "VERSION 5.5")
    fid.write("%s\n" % "bypass no")
    if cascade:
        fid.write("%s\n\n" % "cascade yes")
    else:
        fid.write("%s\n\n" % "cascade no")

    # Write hub and casing lines (channel definition)
    fid.write("%s\n" % "NI_BEGIN CHANNEL")

    # Build the hub and casing line out of basic curves
    # Start the data definition
    fid.write("%s\n" % "NI_BEGIN basic_curve")
    fid.write("%s\n" % "NAME thehub")
    fid.write("%s %i\n" % ("DISCRETISATION", 10))
    fid.write("%s %i\n" % ("DATA_REDUCTION", 0))
    fid.write("%s\n" % "NI_BEGIN zrcurve")
    fid.write("%s\n" % "ZR")

    # Write the length of hub line
    fid.write("%i\n" % ni_h)

    # Write all the points in x,r
    for i in range(ni_h):
        fid.write("%1.11f\t%1.11f\n" % tuple(h[i, :]))

    fid.write("%s\n" % "NI_END zrcurve")
    fid.write("%s\n" % "NI_END basic_curve")

    # Now basic curve for shroud
    fid.write("%s\n" % "NI_BEGIN basic_curve")
    fid.write("%s\n" % "NAME theshroud")

    fid.write("%s %i\n" % ("DISCRETISATION", 10))
    fid.write("%s %i\n" % ("DATA_REDUCTION", 0))
    fid.write("%s\n" % "NI_BEGIN zrcurve")
    fid.write("%s\n" % "ZR")

    # Write the length of shroud
    fid.write("%i\n" % ni_c)

    # Write all the points in x,r
    for i in range(ni_c):
        fid.write("%1.11f\t%1.11f\n" % tuple(c[i, :]))

    fid.write("%s\n" % "NI_END zrcurve")
    fid.write("%s\n" % "NI_END basic_curve")

    # Now lay out the real shroud and hub using the basic curves
    fid.write("%s\n" % "NI_BEGIN channel_curve hub")
    fid.write("%s\n" % "NAME hub")
    fid.write("%s\n" % "VERTEX CURVE_P thehub 0")
    fid.write("%s\n" % "VERTEX CURVE_P thehub 1")
    fid.write("%s\n" % "NI_END channel_curve hub")

    fid.write("%s\n" % "NI_BEGIN channel_curve shroud")
    fid.write("%s\n" % "NAME shroud")
    fid.write("%s\n" % "VERTEX CURVE_P theshroud 0")
    fid.write("%s\n" % "VERTEX CURVE_P theshroud 1")
    fid.write("%s\n" % "NI_END channel_curve shroud")

    fid.write("%s\n" % "NI_END CHANNEL")

    # CHANNEL STUFF DONE
    # NOW DEFINE ROWS
    for i in range(n_row):
        fid.write("%s\n" % "NI_BEGIN nirow")
        fid.write("%s%i\n" % ("NAME r", i + 1))
        fid.write("%s\n" % "TYPE normal")
        fid.write("%s %f\n" % ("PERIODICITY", nb[i]))
        fid.write("%s %i\n" % ("ROTATION_SPEED", 0))

        hdr = [
            "NI_BEGIN NINonAxiSurfaces hub",
            "NAME non axisymmetric hub",
            "REPETITION 0",
            "NI_END   NINonAxiSurfaces hub",
            "NI_BEGIN NINonAxiSurfaces shroud",
            "NAME non axisymmetric shroud",
            "REPETITION 0",
            "NI_END   NINonAxiSurfaces shroud",
            "NI_BEGIN NINonAxiSurfaces tip_gap",
            "NAME non axisymmetric tip gap",
            "REPETITION 0",
            "NI_END   NINonAxiSurfaces tip_gap",
        ]

        fid.writelines("%s\n" % l for l in hdr)

        fid.write("%s\n" % "NI_BEGIN NIBlade")
        fid.write("%s\n" % "NAME Main Blade")

        try:
            if tips[i][0]:
                fid.write("%s\n" % "NI_BEGIN NITipGap")
                fid.write("%s %f\n" % ("WIDTH_AT_LEADING_EDGE", tips[i][0]))
                fid.write("%s %f\n" % ("WIDTH_AT_TRAILING_EDGE", tips[i][1]))
                fid.write("%s\n" % "NI_END NITipGap")
        except TypeError:
            pass

        fid.write("%s\n" % "NI_BEGIN nibladegeometry")
        fid.write("%s\n" % "TYPE GEOMTURBO")
        fid.write("%s\n" % "GEOMETRY_MODIFIED 0")
        fid.write("%s\n" % "GEOMETRY TURBO VERSION 5")
        fid.write("%s %f\n" % ("blade_expansion_factor_hub", 0.1))
        fid.write("%s %f\n" % ("blade_expansion_factor_shroud", 0.1))
        fid.write("%s %i\n" % ("intersection_npts", 10))
        fid.write("%s %i\n" % ("intersection_control", 1))
        fid.write("%s %i\n" % ("data_reduction", 0))
        fid.write("%s %f\n" % ("data_reduction_spacing_tolerance", 1e-006))
        fid.write(
            "%s\n"
            % (
                "control_points_distribution "
                "0 9 77 9 50 0.00622408226922942 0.119480980447523"
            )
        )
        fid.write("%s %i\n" % ("units", 1))
        fid.write("%s %i\n" % ("number_of_blades", 1))

        fid.write("%s\n" % "suction")
        fid.write("%s\n" % "SECTIONAL")
        fid.write("%i\n" % n_sect[i])
        for k in range(n_sect[i]):
            fid.write("%s %i\n" % ("# section", k + 1))
            if cascade:
                fid.write("%s\n" % "XYZ")
            else:
                fid.write("%s\n" % "ZRTH")
            fid.write("%i\n" % ni_ss[i][k])
            for j in range(ni_ss[i][k]):
                fid.write("%1.11f\t%1.11f\t%1.11f\n" % tuple(ss[i][k][j, :]))

        fid.write("%s\n" % "pressure")
        fid.write("%s\n" % "SECTIONAL")
        fid.write("%i\n" % n_sect[i])
        for k in range(n_sect[i]):
            fid.write("%s %i\n" % ("# section", k + 1))
            if cascade:
                fid.write("%s\n" % "XYZ")
            else:
                fid.write("%s\n" % "ZRTH")
            fid.write("%i\n" % ni_ps[i][k])
            for j in range(ni_ps[i][k]):
                fid.write("%1.11f\t%1.11f\t%1.11f\n" % tuple(ps[i][k][j, :]))
        fid.write("%s\n" % "NI_END nibladegeometry")

        # choose a leading and trailing edge treatment

        #    fid.write('%s\n' % 'BLUNT_AT_LEADING_EDGE')
        fid.write("%s\n" % "BLENT_AT_LEADING_EDGE")
        #    fid.write('%s\n' % 'BLENT_TREATMENT_AT_TRAILING_EDGE')
        fid.write("%s\n" % "NI_END NIBlade")

        fid.write("%s\n" % "NI_END nirow")

    fid.write("%s\n" % "NI_END GEOMTURBO")

    fid.close()


def _ssh_env_str():
    return " ".join(["%s=%s" % (v, os.environ[v]) for v in SSH_ENV_VARS])


def _scp_to_remote(to_path, from_path):
    cmd_str = "ssh -q login-e-4 %s scp -q %s %s:%s" % (
        _ssh_env_str(),
        from_path,
        REMOTE,
        to_path,
    )
    os.system(cmd_str)


def _scp_from_remote(to_path, from_path):
    cmd_str = "ssh -q login-e-4 %s scp -q %s:%s %s" % (
        _ssh_env_str(),
        REMOTE,
        from_path,
        to_path,
    )
    return os.WEXITSTATUS(os.system(cmd_str))


def _execute_on_remote(cmd):
    cmd_str = "ssh -q login-e-4 \"%s ssh -q %s '%s'\"" % (
        _ssh_env_str(),
        REMOTE,
        cmd,
    )
    result = subprocess.check_output(cmd_str, shell=True)
    return result


def run_remote(geomturbo, py_scripts, sh_script, gbcs_output_dir):
    """Copy a geomturbo file to gp-111 and run autogrid using scripts."""

    # Try to avoid races when multiple jobs running
    sleep(np.random.rand() * 5.0)

    # Fail if there is not a spare igg licence
    nigg = int(_execute_on_remote("~/bin/num_igg_running.sh").splitlines()[0])
    if nigg >= 3:
        raise RuntimeError("No spare IGG license")

    # Make tmp dir on remote
    tmpdir = _execute_on_remote("mktemp -p ~/tmp/ -d").splitlines()[0]

    # Copy files across
    files = [geomturbo] + py_scripts + [sh_script]
    paths = [os.path.join(os.getcwd(), f) for f in files]
    _scp_to_remote(tmpdir, " ".join(paths))

    # Run the shell script
    _execute_on_remote("cd %s ; bash %s" % (tmpdir, sh_script))

    # Copy mesh back
    remote_mesh_files = (
        "{"
        + ",".join(
            [os.path.join(tmpdir, "mesh." + ext) for ext in ["g", "bcs"]]
        )
        + "}"
    )
    _scp_from_remote(os.path.abspath(gbcs_output_dir), remote_mesh_files)

    # Delete the temporary directory
    if tmpdir.startswith("/home/jb753/tmp"):
        _execute_on_remote("rm -r %s" % tmpdir)
    else:
        pass
        # raise Exception("Trying to delete unexpected %s" % tmpdir)

    # Check the g and bcs arrived
    local_mesh_files = ["mesh.g", "mesh.bcs"]
    success = True
    for f in local_mesh_files:
        fpath = os.path.join(os.path.abspath(gbcs_output_dir), f)
        if not os.path.exists(fpath):
            success = False

    return success


def get_sections_and_annulus_lines(dx_c, rm, Dr, cx, s, tau_c, sect_generators):

    # Distribute the spacings between stator and rotor
    dx_c = np.array([[dx_c[0], dx_c[1] / 2.0], [dx_c[1] / 2.0, dx_c[2]]])

    # Streamwise grids for stator and rotor
    x_c, ilte = zip(*[hmesh.streamwise_grid(dx_ci) for dx_ci in dx_c])
    x = [x_ci * cxi for x_ci, cxi in zip(x_c, cx)]

    # Generate radial grid
    Dr4 = np.stack((Dr[:2], Dr[1:]))
    r = [hmesh.merid_grid(x_ci, rm, Dri) for x_ci, Dri in zip(x_c, Dr4)]
    nr = np.shape(r[0])[1]

    # hub and casing lines
    rh = [ri[:, 0] for ri in r]
    rc = [ri[:, -1] for ri in r]

    # Offset the rotor so it is downstream of stator
    x_offset = x[0][-1] - x[1][0]
    x[1] = x[1] + x_offset

    # Now assemble data for Autogrid
    ps = []
    ss = []
    So_cx = np.empty((2, nr))
    for i in range(2):
        ps.append([])
        ss.append([])
        ile, ite = ilte[i]
        for j in range(nr):
            rnow = r[i][ile : (ite + 1), j]
            spf_now = (rnow[0] - rh[i][ile]) / (rc[i][ile] - rh[i][ile])
            x_cnow = x_c[i][ile : (ite + 1)]
            radial_sect = sect_generators[i](spf_now)
            So_cx[i, j] = geometry._surface_length(
                np.expand_dims(radial_sect, 0)
            )
            xmax = np.max(radial_sect[:, 0, :])
            for side, xrt in zip([ps, ss], radial_sect):
                r_interp = np.interp(xrt[0], x_cnow, rnow)
                xrt *= cx[i] / xmax
                if i > 0:
                    xrt[0] += x_offset
                xrrt = np.insert(xrt, 1, r_interp, 0)
                side[-1].append(xrrt.T)

    # Hub and casing lines in AG format
    x[0] = x[0][:-1]
    rc[0] = rc[0][:-1]
    rh[0] = rh[0][:-1]

    h = np.concatenate(
        [np.column_stack((xi, rhi)) for xi, rhi in zip(x, rh)], 0
    )
    c = np.concatenate(
        [np.column_stack((xi, rci)) for xi, rci in zip(x, rc)], 0
    )

    # Determine number of blades and angular pitch
    nb = np.round(2.0 * np.pi * rm / s / np.mean(So_cx, axis=1)).astype(
        int
    )  # Nearest whole number

    tips = [None, tau_c * cx[1] * np.ones((2,))]

    return x, ilte, nb, ps, ss, h, c, tips


def make_g_bcs(dx_c, rm, Dr, cx, s, tau_c, sect_generators):
    """Generate an OH-mesh for a turbine stage."""

    x, ilte, nb, ps, ss, h, c, tips = get_sections_and_annulus_lines(
        dx_c, rm, Dr, cx, s, tau_c, sect_generators
    )

    write_geomturbo("mesh.geomTurbo", ps, ss, h, c, nb, tips)

    # Do autogrid mesh
    TURBIGEN_ROOT = "/".join(__file__.split("/")[:-2])

    for f in ["script_ag.py2", "script_igg.py2", "script_sh"]:
        shutil.copy(os.path.join(TURBIGEN_ROOT, "ag_mesh", f), ".")

    ntry = 180
    retries = 0
    success = False
    while retries < ntry:

        try:

            success = run_remote(
                "mesh.geomTurbo",
                ["script_ag.py2", "script_igg.py2"],
                "script_sh",
                ".",
            )

        except (subprocess.CalledProcessError, RuntimeError):
            pass

        if success:
            break
        else:
            retries += 1
            print("** Autogrid failed try %d/%d" % (retries, ntry))
            sleep(60.0)

    if not success:
        raise Exception("Autogrid failed")

    return x, ilte, nb
