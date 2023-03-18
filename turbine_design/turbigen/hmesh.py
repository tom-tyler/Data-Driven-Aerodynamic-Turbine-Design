"""Functions to produce a H-mesh from stage design."""
import numpy as np
from . import geometry

# Configure numbers of points
nxb = 81  # Blade chord
nr = 81  # Span
nrt = 65  # Pitch
# nrt = 73  # Pitch

nr_casc = 4  # Radial points in cascade mode
rate = 0.5  # Axial chords required to fully relax
dxsmth_c = 0.1  # Distance over which to fillet shroud corners
tte = 0.04  # Trailing edge thickness


def distribute_spacing(dx_c):
    return np.array([[dx_c[0], dx_c[1] / 2.0], [dx_c[1] / 2.0, dx_c[2]]])


def scale_grid(x_c, cx):
    try:
        return [x_ci * cxi for x_ci, cxi in zip(x_c, cx)]
    except:
        return x_c * cx


def offset_streamwise_grid(x):
    x_offset = x[0][-1] - x[1][0]
    x[1] = x[1] + x_offset
    return x, x_offset


def annulus_lines(r):
    # hub and casing lines
    rh = [ri[:, 0] for ri in r]
    rc = [ri[:, -1] for ri in r]
    return rh, rc


def dimensional_merid_grid(dx_c, rm, Dr, cx):

    # Distribute the spacings between stator and rotor

    # Streamwise grids for stator and rotor
    x_c, ilte = streamwise_grid(dx_c)
    x = x_c * cx

    # Generate radial grid
    r = merid_grid(x_c, rm, Dr)

    # hub and casing lines
    rh, rc = annulus_lines(r)

    return x, ilte, r, rh, rc


def streamwise_grid(dx_c):
    """Generate non-dimensional streamwise grid vector for a blade row.

    The first step in generating an H-mesh is to lay out a vector of axial
    coordinates --- all grid points at a given streamwise index are at the same
    axial coordinate.  Fix the number of points across the blade chord,
    clustered towards the leading and trailing edges. The clustering is then
    mirrored up- and downstream of the row. If the boundary of the row is
    within half a chord of the leading or trailing edges, the clustering is
    truncated. Otherwise, the grid is extendend with constant cell size the
    requested distance.

    The coordinate system origin is the row leading edge. The coordinates are
    normalised by the chord such that the trailing edge is at unity distance.

    Parameters
    ----------
    dx_c: (2,) array [--]
        Distances to row inlet and exit planes, normalised by axial chord

    Returns
    -------
    x_c: (nx,) array [--]
        Streamwise grid vector or vectors, normalised by axial chord.
    ilete: (2,) array [--]
        Indices into x_c for the blade leading and trailing edges.

    """

    clust = geometry.cluster_wall_solve_ER(nxb, 0.001)
    dclust = np.diff(clust)
    dmax = dclust.max()

    # Stretch clustering outside of blade row
    nxb2 = nxb // 2  # Blade semi-chord
    x_c = clust + 0.0  # Make a copy of clustering function
    x_c = np.insert(x_c[1:], 0, clust[nxb2:] - 1.0)  # In front of LE
    x_c = np.append(x_c[:-1], x_c[-1] + clust[: nxb2 + 1])  # Behind TE

    # Numbers of points in inlet/outlet
    # Half a chord subtracted to allow for mesh stretching from LE/TE
    # N.B. Can be negative if we are going to truncate later
    nxu, nxd = [int((dx_ci - 0.5) / dmax) for dx_ci in dx_c]

    if nxu > 0:
        # Inlet extend inlet if needed
        x_c = np.insert(x_c[1:], 0, np.linspace(-dx_c[0], x_c[0], nxu))
    else:
        # Otherwise truncate and rescale so that inlet is in exact spot
        x_c = x_c[x_c > -dx_c[0]]
        x_c[x_c < 0.0] = x_c[x_c < 0.0] * -dx_c[0] / x_c[0]
    if nxd > 0:
        # Outlet extend if needed
        x_c = np.append(x_c[:-1], np.linspace(x_c[-1], dx_c[1] + 1.0, nxd))
    else:
        # Otherwise truncate and rescale so that outlet is in exact spot
        x_c = x_c[x_c < dx_c[1] + 1.0]
        x_c[x_c > 1.0] = (x_c[x_c > 1.0] - 1.0) * dx_c[1] / (
            x_c[-1] - 1.0
        ) + 1.0

    # Get indices of leading and trailing edges
    # These are needed later for patching
    i_edge = [np.where(x_c == xloc)[0][0] for xloc in [0.0, 1.0]]

    return x_c, i_edge


def merid_grid(x_c, rm, Dr):
    """Generate meridional grid for a blade row.

    Each spanwise grid index corresponds to a surface of revolution. So the
    gridlines have the same :math:`(x, r)` meridional locations across the
    entire row pitch.

    Parameters
    ----------
    x_c: (nx,) array [--]
        Streamwise grid vector normalised by axial chord.
    rm: float [m]
        A constant mean radius for this blade row or rows.
    Dr: (2,) array [m]
        Annulus spans at inlet and exit of blade row.

    Returns
    -------
    r : (nx, nr) or (nrow, nx, nr) array [m]
        Radial coordinates for each point in the meridional view.

    """

    # Evaluate hub and casing lines on the streamwise grid vector
    # Linear between leading and trailing edges, defaults to constant outside
    rh = np.interp(x_c, [0.0, 1.0], rm - Dr / 2.0)
    rc = np.interp(x_c, [0.0, 1.0], rm + Dr / 2.0)

    # Smooth the corners over a prescribed distance
    geometry.fillet(x_c, rh, dxsmth_c)  # Leading edge around 0
    geometry.fillet(x_c - 1.0, rc, dxsmth_c)  # Trailing edge about 1

    # Check htr to decide if this is a cascade
    htr = rh[0] / rc[0]
    if htr > 0.95:
        # Define a uniform span fraction row vector
        spf = np.atleast_2d(np.linspace(0.0, 1.0, nr_casc))
    else:
        # Define a clustered span fraction row vector
        spf = np.atleast_2d(geometry.cluster_cosine(nr))

    # Evaluate radial coordinates: dim 0 is streamwise, dim 1 is radial
    r = spf * np.atleast_2d(rc).T + (1.0 - spf) * np.atleast_2d(rh).T

    return r


def b2b_grid(x, ilte, r, cx, sect, s):
    """Generate circumferential coordinates for a blade row."""

    ni = len(x)
    nj = r.shape[1]
    nk = nrt

    ile, ite = ilte

    # Dimensional axial coordinates
    x = np.reshape(x, (-1, 1, 1))
    r = np.atleast_3d(r)

    x_c = x / cx

    # Determine number of blades and angular pitch
    r_m = np.mean(r[0, (0, -1), 0])
    nblade = np.round(2.0 * np.pi * r_m / s)  # Nearest whole number
    pitch_t = 2 * np.pi / nblade

    # Preallocate and loop over radial stations
    rtlim = np.nan * np.ones((ni, nj, 2))
    for j in range(nj):

        # Retrieve blade section as [surf, x or y, index]
        loop_xrt = sect[j]

        # Shift the leading edge to first index
        loop_xrt = np.roll(loop_xrt, -np.argmin(loop_xrt[0]), axis=1)

        # Now split the loop back up based on true LE/TE
        ile = np.argmin(loop_xrt[0])
        ite = np.argmax(loop_xrt[0])
        upper_xrt = loop_xrt[:, ile : (ite + 1)]
        lower_xrt = np.insert(
            np.flip(loop_xrt[:, ite:-1], -1), 0, loop_xrt[:, ile], -1
        )

        # Interpolate the pitchwise limits
        rtlim[:, j, 0] = np.interp(x[:, 0, 0], *upper_xrt)
        rtlim[:, j, 1] = (
            np.interp(x[:, 0, 0], *lower_xrt) + pitch_t * r[:, j, 0]
        )

    # Define a pitchwise clustering function with correct dimensions
    clust = geometry.cluster_cosine(nk).reshape(1, 1, -1)

    # Relax clustering towards a uniform distribution at inlet and exit
    # With a fixed ramp rate
    unif_rt = np.linspace(0.0, 1.0, nk).reshape(1, 1, -1)
    relax = np.ones_like(x_c)
    relax[x_c < 0.0] = 1.0 + x_c[x_c < 0.0] / rate
    relax[x_c > 1.0] = 1.0 - (x_c[x_c > 1.0] - 1.0) / rate
    relax[relax < 0.0] = 0.0
    clust = relax * clust + (1.0 - relax) * unif_rt

    # Fill in the intermediate pitchwise points using clustering function
    rt = rtlim[..., (0,)] + np.diff(rtlim, 1, 2) * clust

    return rt


def scale_section(side_x_rt, cx):
    xmax = np.max(side_x_rt[:, 0, :])
    return side_x_rt * cx / xmax


def row_grid(dx_cx, cx, rm, Dr, s_So, sect_generator):
    """H-mesh for one row.

    Parameters
    ----------
    dx_cx: (2,) float array [--]
        Distances to row inlet and outlet as numbers of axial chords.
    cx: float [m]
        Aerofoil axial chord.
    rm: float [m]
        Annulus mean radius.
    Dr: (2,) float array [m]
        Annulus heights at row inlet and exit.
    s_So: float [m]
        Blade pitch to suction surface length ratio.
    sect_generator: callable
        Function that takes span fraction as argument, returns side_xrt.

    Returns
    --------
    x, r, rt, ilte

    """

    # Ensure inputs are 1d numpy arrays
    dx_cx = np.reshape(dx_cx, -1)
    Dr = np.reshape(Dr, -1)

    # Make streamwise grid
    x_cx, ilte = streamwise_grid(dx_cx)

    # Scale by chord
    x = x_cx * cx

    # Generate radial grid
    r = merid_grid(x_cx, rm, Dr)

    # Evaluate blade sections at query span fractions given by radial grid
    rle = r[ilte[0], :]
    spf = (rle - rle.min()) / rle.ptp()

    # Get sections (normalised by axial chord for now)
    nr = r.shape[1]
    sect = []
    So_cx = np.empty_like(rle)
    for j in range(nr):
        sect_now = sect_generator(spf[j])
        So_cx[j] = geometry._surface_length(np.expand_dims(sect_now, 0))
        sect.append(geometry._loop_section(scale_section(sect_now, cx)))

    # Blade pitch
    s = np.mean(s_So * So_cx * cx)

    # Now we can do b2b grid
    rt = b2b_grid(x, ilte, r, cx, sect, s)

    return x, r, rt, ilte
