import scipy.optimize
import scipy.integrate
from . import compflow, util
import numpy as np

expon = 0.62
muref = 1.8e-5
Tref = 288.0


# Define a namedtuple to store all information about a non-dim stage design
fan_vars = {
    "Yp": r"Stagnation pressure loss coefficient :math:`Y_p` [--]",
    "Ma": r"Mach numbers :math:`\Ma` [--]",
    "Marel": r"Rotor-relative Mach numbers :math:`\Ma^\rel` [--]",
    "Al": r"Yaw angles :math:`\alpha` [deg]",
    "Alrel": r"Rotor-relative yaw angles :math:`\alpha^\rel` [deg]",
    "Ax_Ax1": r"Annulus area ratios :math:`A_x/A_{x1}` [--]",
    "Mab": r"Blade Mach number :math:`U/\sqrt{\gamma R T}` [--]",
    "Po_Po1": r"Stagnation pressure ratios :math:`p_0/p_{01}` [--]",
    "To_To1": r"Stagnation temperature ratios :math:`T_0/T_{01}` [--]",
    "U_sqrt_cpTo1": r"Non-dimensional blade speed :math:`U/\sqrt{c_p T_{01}}` [--]",
    "ga": r"Ratio of specific heats, :math:`\gamma` [--]",
    "phi": r"Flow coefficient, :math:`\phi` [--]",
    "Psi": r"Total-to-total pressure rise coefficient, :math:`\Psi` [--]",
    "Psi_ts": r"Total-to-statis pressure rise coefficient, :math:`\Psi_\mathrm{ts}` [--]",
    "Vt_U": r"Normalised tangential velocities, :math:`V_\theta/U` [--]",
    "Vtrel_U": r"Normalised relative tangential velocities, :math:`V^\rel_\theta/U` [--]",
    "V_U": r"Normalised velocities, :math:`V/U` [--]",
    "Vrel_U": r"Normalised relative velocities, :math:`V/U` [--]",
    "P_Po1": r"Total-to-static pressure ratios, :math:`p/p_{01}` [--]",
    "T_To1": r"Total-to-static temperature ratios, :math:`T/T_{01}` [--]",
    "eta": r"Polytropic efficiency, :math:`\eta` [--]",
}
docstring_NonDimFan = (
    "Data class to hold non-dimensional geometry and derived flow parameters "
    "of a single-row fan mean-line design."
)
NonDimFan = util.make_namedtuple_with_docstrings(
    "NonDimFan", docstring_NonDimFan, fan_vars
)


def nondim_fan_total_static(
    phi,  # Flow coefficient [--]
    Psi_ts,  # Total-to-static pressure rise coefficient [--]
    Al1,  # Inlet yaw angle [deg]
    Mab,  # Blade Mach number [--]
    ga,  # Ratio of specific heats [--]
    eta,  # Polytropic efficiency [--]
    Vx_rat=1.0,  # Axial velocity ratio [--]
):
    r"""Get geometry for a rotor-only fan using total-to-static pressure rise."""

    # Iteration step: returns error in TS rise as function of TT rise
    def iter_Psi(x):
        fan_now = nondim_fan_total_total(phi, x, Al1, Mab, ga, eta, Vx_rat)
        return fan_now.Psi_ts - Psi_ts

    # Find root
    Psi_soln = scipy.optimize.root_scalar(iter_Psi, x0=1e-6, x1=10.0).root

    # Once we have a solution, evaluate stage geometry
    fan_out = nondim_fan_total_total(phi, Psi_soln, Al1, Mab, ga, eta, Vx_rat)

    return fan_out


def nondim_fan_total_total(
    phi,  # Flow coefficient [--]
    Psi,  # Total-to-total pressure rise coefficient [--]
    Al1,  # Inlet yaw angle [deg]
    Mab,  # Blade Mach number [--]
    ga,  # Ratio of specific heats [--]
    eta,  # Polytropic efficiency [--]
    Vx_rat=1.0,  # Axial velocity ratio [--]
):
    r"""Get geometry for a rotor-only fan using total-to-total pressure rise.

    Parameters
    ----------
    phi : float
        Flow coefficient, :math:`\phi`.
    Psi : float
        Total-to-total pressure rise coefficient, :math:`\Psi`.
    Al1 : array
        Absolute yaw angle at inlet, :math:`\alpha_1`.
    Mab : float
        Blade Mach number, :math:`\Ma_\mathrm{blade}`.
    ga : float
        Ratio of specific heats, :math:`\gamma`.
    eta : float
        Polytropic efficiency, :math:`\eta`.
    Vx_rat : float, default=1.
        Axial velocity ratio, :math:`\zeta`.

    Returns
    -------
    """

    #
    # First, construct velocity triangles
    #

    # Evaluate total pressure ratio
    half_rho_Usq_Po1 = (0.5 * ga * Mab ** 2.0) / compflow.Po_P_from_Ma(Mab, ga)
    Po2_Po1 = 1.0 + Psi * half_rho_Usq_Po1

    # Polytropic effy to get temperature ratio
    To2_To1 = Po2_Po1 ** ((ga - 1.0) / ga / eta)

    # Use Euler work equation to get exit absolute flow angle
    Usq_cpTo1 = compflow.V_cpTo_from_Ma(Mab, ga) ** 2.0
    tanAl1 = np.tan(np.radians(Al1))
    tanAl2 = ((To2_To1 - 1.0) / phi / Usq_cpTo1 + tanAl1) / Vx_rat
    Al2 = np.degrees(np.arctan(tanAl2))

    # Now we have all flow angles
    Al = np.array([Al1, Al2])
    Alrad = np.radians(Al)
    cosAl = np.cos(Alrad)

    # Get non-dimensional velocities from definition of flow coefficient
    Vx_U = np.array([1.0, Vx_rat]) * phi
    Vt_U = Vx_U * np.tan(Alrad)
    V_U = np.sqrt(Vx_U ** 2.0 + Vt_U ** 2.0)

    # Change reference frame for rotor-relative velocities and angles
    Vtrel_U = Vt_U - 1.0
    Vrel_U = np.sqrt(Vx_U ** 2.0 + Vtrel_U ** 2.0)
    Alrel = np.degrees(np.arctan2(Vtrel_U, Vx_U))

    # Non-dimensional temperatures from blade Ma and temperature ratio
    To_To1 = np.array([1.0, To2_To1])
    cpTo_Usq = To_To1 / Usq_cpTo1

    # Mach numbers and capacity from compressible flow relations
    Ma = compflow.Ma_from_V_cpTo(V_U / np.sqrt(cpTo_Usq), ga)
    Marel = Ma * Vrel_U / V_U
    Q = compflow.mcpTo_APo_from_Ma(Ma, ga)
    Q_Q1 = Q / Q[0]

    #
    # Second, construct annulus line
    #

    # Convert to stagnation pressures
    Po_Po1 = np.array([1.0, Po2_Po1])

    # Use definition of capacity to get flow area ratios
    # Area ratios = span ratios because rm = const
    Dr_Drin = np.sqrt(To_To1) / Po_Po1 / Q_Q1 * cosAl[0] / cosAl

    # Evaluate some other useful secondary aerodynamic parameters
    T_To1 = To_To1 / compflow.To_T_from_Ma(Ma, ga)
    P_Po1 = Po_Po1 / compflow.Po_P_from_Ma(Ma, ga)
    Porel_Po1 = P_Po1 * compflow.Po_P_from_Ma(Marel, ga)

    # Stagnation pressure loss coefficients
    Yp = (Porel_Po1[0] - Porel_Po1[1]) / (Porel_Po1[0] - P_Po1[0])

    # Total-to-static pressure rise coefficient
    Psi_ts = (P_Po1[-1] - 1.0) / half_rho_Usq_Po1

    # Assemble all of the data into the output object
    fan = NonDimFan(
        Yp=Yp,
        Al=Al,
        Alrel=Alrel,
        Ma=Ma,
        Marel=Marel,
        Ax_Ax1=Dr_Drin,
        Mab=Mab,
        Po_Po1=Po_Po1,
        To_To1=To_To1,
        phi=phi,
        Psi=Psi,
        U_sqrt_cpTo1=np.sqrt(Usq_cpTo1),
        ga=ga,
        Vt_U=Vt_U,
        Vtrel_U=Vtrel_U,
        V_U=V_U,
        Vrel_U=Vrel_U,
        P_Po1=P_Po1,
        T_To1=T_To1,
        eta=eta,
        Psi_ts=Psi_ts,
    )

    return fan
