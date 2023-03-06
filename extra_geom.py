import numpy as np
from scipy.special import binom
from numpy.linalg import lstsq
from scipy.optimize import newton, fminbound
from scipy.interpolate import interp1d

nx=101

def cluster_cosine(npts):
    """Return a cosinusoidal clustering function with a set number of points."""
    # Define a non-dimensional clustering function
    return 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, npts)))
        
def _prelim_thickness(x, tte=0.04, xtmax=0.2, tmax=0.15):
    """A rough cubic thickness distribution."""
    tle = tte / 2.0
    tlin = tle + x * (tte - tle)
    # Additional component to make up maximum thickness
    tmod = tmax - xtmax * (tte - tle) - tle
    # Total thickness
    thick = tlin + tmod * (
        1.0 - 4.0 * np.abs(x ** (np.log(0.5) / np.log(xtmax)) - 0.5) ** 2.0
    )
    return thick

def _set_camber_line_exponent(tanchi, tangam):
    """Choose exponent for power law camber line with awkward special cases."""
    if tangam == 0.0:
        n = 1.0
    else:
        n = np.clip(np.sum(tanchi) / tangam, 1.0, 20.0)
    return n


def evaluate_camber_slope(x, chi, stag):
    """Camber line slope as a function of x, given inlet and exit angles."""
    tanchi = np.tan(np.radians(chi))
    tangam = np.tan(np.radians(stag))
    n = _set_camber_line_exponent(tanchi, tangam)
    a = tanchi[1] / n
    b = -tanchi[0] / n
    dy = a * n * x ** (n - 1.0) - b * n * (1.0 - x) ** (n - 1.0)
    return dy

def evaluate_camber(x, chi, stag):
    """Camber line as a function of x, given inlet and exit angles."""
    tanchi = np.tan(np.radians(chi))
    tangam = np.tan(np.radians(stag))
    n = _set_camber_line_exponent(tanchi, tangam)
    a = tanchi[1] / n
    b = -tanchi[0] / n
    y = a * x ** n + b * (1.0 - x) ** n
    y = y - y[0]

    return y


def _thickness_to_coord(xc, t, chi, stag):
    theta = np.arctan(evaluate_camber_slope(xc, chi, stag))
    yc = evaluate_camber(xc, chi, stag)
    xu = xc - t * np.sin(theta)
    yu = yc + t * np.cos(theta)
    return xu, yu

def _bernstein(x, n, i):
    """Evaluate ith Bernstein polynomial of degree n at some x-coordinates."""
    return binom(n, i) * x ** i * (1.0 - x) ** (n - i)


def _to_shape_space(x, z, zte):
    """Transform real thickness to shape space."""
    # Ignore singularities at leading and trailing edges
    eps = 1e-6
    with np.errstate(invalid="ignore", divide="ignore"):
        ii = np.abs(x - 0.5) < (0.5 - eps)
    s = np.ones(x.shape) * np.nan
    s[ii] = (z[ii] - x[ii] * zte) / (np.sqrt(x[ii]) * (1.0 - x[ii]))
    # s[ii] = z[ii] / (np.sqrt(x[ii]) * np.sqrt(1.0 - x[ii]))
    return s

def _coord_to_thickness(xy, chi, stag):
    """Perpendicular thickness distribution given camber line angles.

    Parameters
    ----------

    xy : array, 2-by-...
        Cartesian coordinates of a blade surface.
    chi : array, len 2
        Camber angles for inlet and outlet."""

    # Split into x and y
    x, y = xy
    # Find intersections of xu, yu with camber line perpendicular
    def iterate(xi):
        return (
            y
            - evaluate_camber(xi, chi, stag)
            + (x - xi) / evaluate_camber_slope(xi, chi, stag)
            # This is sufficient for 101 points along chord
        )

    with np.errstate(invalid="ignore", divide="ignore"):
        xc = newton(iterate, x)
    yc = evaluate_camber(xc, chi, stag)
    # Now evaluate thickness
    t = np.sqrt(np.sum(np.stack((x - xc, y - yc), axis=0) ** 2.0, axis=0))

    return xc, yc, t


def _fit_aerofoil(xy, chi, stag, order):
    """Fit Bernstein polynomials to both aerofoil surfaces simultaneously."""
    n = order - 1
    # When converting from real coordinates to shape space, we end up with
    # singularities and numerical instability at leading and trailing edges.
    # So in these cases, ignore within dx at LE and TE
    dx = 0.02
    xtrim_all = []
    strim_all = []
    X_all = []
    X_le_all = []
    for xyi in xy:
        xc, yc, t = _coord_to_thickness(xyi, chi, stag)
        s = _to_shape_space(xc, t, 0.02)
        with np.errstate(invalid="ignore", divide="ignore"):
            itrim = np.abs(xc - 0.5) < (0.5 - dx)
        xtrim_all.append(xc[itrim])
        strim_all.append(s[itrim])
        X_all.append(
            np.stack([_bernstein(xc[itrim], n, i) for i in range(1, n + 1)]).T
        )
        X_le_all.append(_bernstein(xc[itrim], n, 0))

    strim = np.concatenate(strim_all)
    X_le = np.concatenate(X_le_all)
    X = np.block(
        [
            [X_all[0], np.zeros(X_all[1].shape)],
            [np.zeros(X_all[0].shape), X_all[1]],
        ]
    )
    X = np.insert(X, 0, X_le, 1)
    A_all, resid = lstsq(X, strim, rcond=None)[:2]
    Au = A_all[:order]
    Al = np.insert(A_all[order:], 0, A_all[0])
    return np.vstack((Au, Al)), resid


def prelim_A():
    """Get values of A corresponding to preliminary thickness distribution."""

    xc = cluster_cosine(nx)

    # Camber line
    thick = _prelim_thickness(xc)

    # Choose arbitrary camber line (A independent of chi)
    chi = (-10.0, 20.0)

    stagger = np.mean(chi)

    # Assemble preliminary upper and lower coordiates
    xy_prelim = [
        _thickness_to_coord(xc, sgn * thick, chi, stagger) for sgn in [1, -1]
    ]

    # Fit Bernstein polynomials to the prelim coords in shape space
    A, _ = _fit_aerofoil(xy_prelim, chi, stagger, order=4)

    return A