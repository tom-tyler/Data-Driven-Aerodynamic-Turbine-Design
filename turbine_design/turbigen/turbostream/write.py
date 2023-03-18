"""Functions for exporting a stage design to Turbostream."""
import numpy as np
from turbigen import ohmesh, hmesh, util
from . import grid, cooling, warp

muref = 1.8e-5

COOL_SCHEME = cooling.DEFAULT

def add_h_block(g, xin, rin, rtin, ilte):
    """From mesh coordinates, add a block with patches to TS grid object"""

    # Wrestle the coordinates to correct number of dimensions
    ni, nj, nk = np.shape(rtin)
    rt = np.copy(rtin)
    r = np.repeat(rin[:, :, None], nk, axis=2)
    if xin.ndim == 3:
        x = np.copy(xin)
    else:
        x = np.tile(xin[:, None, None], (1, nj, nk))

    # Add new block from these coordinates
    b, bid = g.make_block(x, r, rt)

    ile, ite = ilte

    # Add periodic patches first

    # Upstream of LE
    periodic_up_1, pid_up_1 = g.make_patch(
        kind="periodic",
        bid=bid,
        i=(0, ile + 1),
        j=(0, nj),
        k=(0, 1),
        nxbid=bid,
        dirs=(0, 1, 6),
    )

    periodic_up_2, pid_up_2 = g.make_patch(
        kind="periodic",
        bid=bid,
        i=(0, ile + 1),
        j=(0, nj),
        k=(nk - 1, nk),
        dirs=(0, 1, 6),
        nxbid=bid,
    )
    periodic_up_1.nxpid = pid_up_2
    periodic_up_2.nxpid = pid_up_1

    # Downstream of TE
    periodic_dn_1, pid_dn_1 = g.make_patch(
        kind="periodic",
        bid=bid,
        i=(ite, ni),
        j=(0, nj),
        k=(0, 1),
        dirs=(0, 1, 6),
        nxbid=bid,
    )
    periodic_dn_2, pid_dn_2 = g.make_patch(
        kind="periodic",
        bid=bid,
        i=(ite, ni),
        j=(0, nj),
        k=(nk - 1, nk),
        dirs=(0, 1, 6),
        nxbid=bid,
    )
    periodic_dn_1.nxpid = pid_dn_2
    periodic_dn_2.nxpid = pid_dn_1


def _set_variables(g, avs, mu=None):
    """Set application and block variables on a TS grid object."""

    # Override some block variable defaults
    g.set_bv("fmgrid", 0.2)
    g.set_bv("poisson_fmgrid", 0.0)
    g.set_bv("xllim_free", 0.1)
    g.set_bv("free_turb", 0.05)
    g.set_bv("fsturb", 1.0)
    g.set_bv("fac_st0", 1.0)

    g.set_bv("ktrans_i1_frac", 0.0)
    g.set_bv("ktrans_i2_frac", 0.0)
    g.set_bv("ktrans_j1_frac", 0.0)
    g.set_bv("ktrans_j2_frac", 0.0)

    # Override application variable defaults
    g.set_av("restart", 1)
    g.set_av("ilos", 1)
    g.set_av("use_temperature_sensor", 1)

    g.set_av("sfin", 0.5)
    g.set_av("facsecin", 0.005)
    g.set_av("cfl", 0.4)
    g.set_av("poisson_cfl", 0.7)

    g.set_av("rfmix", 0.01)
    g.set_av("write_yplus", 1)

    g.set_av("smooth_scale_dts_option", 0)

    g.set_av("poisson_restart", 0)
    g.set_av("poisson_nsmooth", 10)
    g.set_av("poisson_sfin", 0.02)
    g.set_av("poisson_nstep", 5000)

    # Deal with viscosity
    if mu:
        # Set a constant viscosity if one is specified
        g.set_av("viscosity", mu)
        g.set_av("viscosity_law", 0)
    else:
        # Otherwise, use power law
        g.set_av("viscosity", muref)
        g.set_av("viscosity_law", 1)

    # No multigrid at mixing planes
    bid_mix, _ = g.find_patches(grid.KIND.mixing)
    g.set_bv("fmgrid", 0.0, bid_mix)

    # Kill turbulence in outlet for better convergence
    bid_out, _ = g.find_patches(grid.KIND.outlet)
    g.set_bv("fac_st0", 0.0, bid_out)

    # Apply settings from dict
    for name, val in avs.items():
        g.set_av(name, val)

    # Average over last few steps
    g.set_av("nstep_save_start", g.get_av("nstep") - 10000)

    # Mixing length limit
    g.apply_xllim(np.ones((2,)) * 0.03)


def _apply_bconds(g, Poin, Toin, delta, Pout):
    ga = g.get_av("ga")

    # Inlet
    spf = np.linspace(0.0, 1.0, 1000)
    Po_Poinf = util.boundary_layer_Po_Poinf(spf, delta, Mainf=0.3, ga=ga)
    g.apply_inlet_1d(spf, Poin * Po_Poinf, Toin, pitch=0.0, yaw=0.0)

    # Outlet
    g.apply_outlet(Pout)

    # Wall rotations
    g.apply_rotation(["stationary", "shroud"])

def _apply_cooling(g, Dstg, cool_scheme):
    """Add cooling to a Turbostream grid."""

    # Scale the cooling scheme to have correct total mass flow in each row
    for row, fc in zip(cool_scheme, Dstg.fc):
        fc_input = np.sum([patch.fc for patch in row])
        scale_factor = fc/fc_input
        for patch in row:
            patch.fc *= scale_factor

    # Get the absolute temperatures and mass flows at row inlets
    Toinf = Dstg.To1 * Dstg.To_To1[:2]
    mdotinf = Dstg.mdot_mdot1 * Dstg.mdot1

    # Convert rel frame coolant To ratio to abs frame
    # Need to check this maths...
    Usq_cpTo = Dstg.U_sqrt_cpTo1 ** 2.0
    rel_fac_inf = (
        0.5
        * Usq_cpTo
        * (1.0 - 2.0 * Dstg.phi * np.tan(np.radians(Dstg.Al[1])))
    )
    rel_fac_cool = 0.5 * Usq_cpTo * (1.0 - 2.0 * Dstg.preswirl_factor)
    To2_To1 = Dstg.To_To1[1] / Dstg.To_To1[0]
    for patch in cool_scheme[1]:
        Toc2_To1 = patch.TR * (To2_To1 + rel_fac_inf) - rel_fac_cool
        Toc2_To2 = Toc2_To1 / Dstg.To_To1[1]
        patch.TR = Toc2_To2

    # Add the patches
    cooling.add_to_grid(cool_scheme, g, mdotinf, Toinf)


def make_h_mesh(x, r, rt, ilte):
    """From coordinates, make a patched H mesh."""
    # Make grid, add the blocks
    g = grid.Grid()

    for args in zip(x, r, rt, ilte):
        add_h_block(g, *args)

    grid.set_default_variables(g)

    # Calculate number of blades
    t = [rti / ri[..., None] for rti, ri in zip(rt, r)]
    nb = [
        np.asscalar(np.round(2.0 * np.pi / np.diff(ti[0, 0, (0, -1)], 1))) for ti in t
    ]
    nb_int = [int(nbi) for nbi in nb]
    for bid, nbi in enumerate(nb_int):
        g.set_bv("nblade", nbi)

    # Make boundary patches
    inlet, _ = g.make_patch(
        kind="inlet",
        bid=0,
        i=(0, 1),
        j=(0, g.nj[0]),
        k=(0, g.nk[0]),
    )
    outlet, _ = g.make_patch(
        kind="outlet",
        bid=1,
        i=(g.ni[1] - 1, g.ni[1]),
        j=(0, g.nj[1]),
        k=(0, g.nk[1]),
    )
    mix_up, pid_mix_up = g.make_patch(
        kind="mixing",
        bid=0,
        i=(g.ni[0] - 1, g.ni[0]),
        j=(0, g.nj[0]),
        k=(0, g.nk[0]),
        nxbid=1,
        dirs=(6, 1, 2),
    )
    mix_dn, pid_mix_dn = g.make_patch(
        kind="mixing",
        bid=1,
        i=(0, 1),
        j=(0, g.nj[1]),
        k=(0, g.nk[1]),
        nxbid=0,
        dirs=(6, 1, 2),
    )
    mix_dn.nxpid = pid_mix_up
    mix_up.nxpid = pid_mix_dn
    return g


def write_stage_from_params(params, fname):
    """Generate a Turbostream input file from a dictionary of parameters."""

    # Set geometry using dimensional bcond and 3D design parameter
    Dstg = params.dimensional_stage
    stg = params.nondimensional_stage

    # Recalculate the boundary layer thickness relative to span
    delta_span = params.delta * Dstg.cx[0] / Dstg.Dr[0]

    # The meshing routines need "Section generators", callables that take a
    # span fraction and output a blade section
    def vane_section(spf):
        return params.interpolate_section(0, spf)

    def blade_section(spf):
        return params.interpolate_section(1, spf, is_rotor=True)

    sect_generators = [vane_section, blade_section]

    grid_type = params.grid_type

    # Choose how to make mesh based on grid type
    if grid_type == "warp":

        print('Warping')
        # Make 
        x, ilte, nb, ps, ss, h, c, tips = ohmesh.get_sections_and_annulus_lines(
            params.dx_c,
            Dstg.rm,
            Dstg.Dr,
            Dstg.cx,
            Dstg.s,
            params.tau_c,
            sect_generators,
        )

        g = warp.warp(params.guess_file, ps, ss)

    elif grid_type == "oh":

        # Make a g and bcs
        print('Sending to Autogrid')
        x, ilte, nb = ohmesh.make_g_bcs(
            params.dx_c,
            Dstg.rm,
            Dstg.Dr,
            Dstg.cx,
            Dstg.s,
            params.tau_c,
            sect_generators,
        )

        # Read from AutoGrid
        g = grid.read_autogrid("mesh.bcs", "mesh.g")

    elif grid_type == "h":

        dx_c = hmesh.distribute_spacing(params.dx_c)

        block_geometry = []
        for i in range(2):

            # Meridional grid first
            block_geometry.append(
                hmesh.row_grid(
                    dx_c[i],
                    Dstg.cx[i],
                    Dstg.rm,
                    Dstg.Dr[i : (i + 2)],
                    Dstg.s_cx[i],
                    sect_generators[i],
                )
            )

        ilte = [bgi[-1] for bgi in block_geometry]
        x = [bgi[0] for bgi in block_geometry]
        r = [bgi[1] for bgi in block_geometry]
        rt = [bgi[2] for bgi in block_geometry]
        x_offset = x[0][-1] - x[1][0]
        x[1] += x_offset

        g = make_h_mesh(*zip(*block_geometry))
        rpm = Dstg.Omega / 2.0 / np.pi * 60.0
        g.set_bv("rpm", 0.0, 0)
        g.set_bv("rpm", rpm, 1)

        t = [rti / np.expand_dims(ri, 2) for ri, rti in zip(r, rt)]
        dt = np.array([np.mean(np.ptp(ti[0, :, :], axis=-1)) for ti in t])
        nblade = np.round(2.0 * np.pi / dt).astype(int)
        g.apply_nblade(nblade)



    # Now prepare the grid

    # Apply application/block variables
    avs = params.cfd_config.copy()
    avs["ga"] = stg.ga
    avs["cp"] = Dstg.cp
    _set_variables(g, avs,mu=Dstg.mu)

    # Damp down problem areas
    if grid_type == "oh":
        fac = 10.0 / 25.0
        g.set_bv("dampin_mul", fac, g.trailing_edge_bids)
        g.set_bv("dampin_mul", fac, g.tip_gap_bids)

    # Boundary conditions
    P3 = stg.P3_Po1 * Dstg.Po1
    _apply_bconds(g, Dstg.Po1, Dstg.To1, delta_span, P3)

    # Cooling flows
    _apply_cooling(g, Dstg, COOL_SCHEME)

    # No rotating hub/casing at exit if OH mesh
    if grid_type == "oh":
        bid_out = g.find_patches(grid.KIND.outlet)[0]
        g.set_bv("rpmj1", 0.0, bid_out)
        g.set_bv("rpmj2", 0.0, bid_out)

    if params.guess_file:

        # Use guess from file
        g.guess_file(params.guess_file)

        # No multigrid or smooth start needed
        g.set_bv("fmgrid", 0.0)

    else:

        # Apply an initial guess by linear interpolation in  x

        # Axial coordinates
        ii0 = np.concatenate(([0], ilte[0]))
        ii1 = np.concatenate((ilte[1], [-1]))
        xg = np.concatenate(
            [
                x[0][ii0],
                x[1][ii1],
            ]
        )

        # Flow variables
        Pog = np.repeat(stg.Po_Po1 * Dstg.Po1, 2)
        Tog = np.repeat(stg.To_To1 * Dstg.To1, 2)
        Mag = np.repeat(stg.Ma, 2)
        Alg = np.repeat(stg.Al, 2)

        # Interpolate to grid
        g.guess_1d(xg, Pog, Tog, Mag, Alg)

    # Load balance and write out
    g.load_balance(1)
    g.write_hdf5(fname)
