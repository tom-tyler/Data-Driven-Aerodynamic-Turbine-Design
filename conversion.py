import numpy as np

def pitch_circulation(stg, C0):
    r"""Calculate pitch-to-chord ratios using circulation coefficient.
    """

    if stg.psi < 0.0:
        V2 = np.array((stg.Vrel_U[1], stg.V_U[2]))
        Vt2 = np.array((stg.Vtrel_U[1], stg.Vt_U[2]))
        Vt1 = np.array((stg.Vtrel_U[0], stg.Vt_U[1]))
    else:
        V2 = np.array((stg.V_U[1], stg.Vrel_U[2]))
        Vt2 = np.array((stg.Vt_U[1], stg.Vtrel_U[2]))
        Vt1 = np.array((stg.Vt_U[0], stg.Vtrel_U[1]))

    return C0 * V2 / np.abs(Vt1 - Vt2)

def nondim_stage_from_Al(
    phi,  # Flow coefficient [--]
    psi,  # Stage loading coefficient [--]
    Al13,  # Yaw angles [deg]
    Ma2,  # Vane exit Mach number [--]
    ga,  # Ratio of specific heats [--]
    eta,  # Polytropic efficiency [--]
    Vx_rat=(1.0, 1.0),  # Axial velocity ratios [--]
    loss_rat=0.5,  # Fraction of stator loss [--]
    mdotc_mdot1=(0.0, 0.0),  # Coolant flows as fraction of inlet [--]
    Toc_Toinf=(1.0, 1.0),  # Local coolant temperature ratios [--]
    preswirl_factor=0.0,  # Preswirl factor [--]
):
    r"""Get geometry for an aerodynamic parameter set specifying outlet swirl.

    This routine calculates the non-dimensional *geometric* parameters that
    correspond to an input set of non-dimensional *aerodynamic* parameters. In
    this way, a turbine designer can directly specify meaningful quantities
    that characterise the desired fluid dynamics while the precise blade
    and annulus geometry are abstracted away.

    The working fluid is a perfect gas obeying the standard compressible flow
    relations. The mean radius, angular velocity, and hence blade speed are
    constant throughout the turbine stage.

    From the output of this function, arbitrarily choosing one of angular
    velocity or mean radius, and providing an inlet stagnation state, will
    completely define the stage in dimensional terms.

    Coolant mass flows are specified as a fraction of the *inlet* mass flow.

    Coolant temperatures are specified as a ratio of coolant stagnation
    temperature to row inlet stagnation temperature, in the relative frame for
    rotors. Converting coolant temperatures to the absolute frame then requires
    knowledge of coolant preswirl. Change in relative stagnation temperature
    due to preswirl is a function of the factor,

    .. math ::

        \newcommand{\PS}{\mathrm{PS}}

        \frac{V_{\theta,\PS} r_\PS}{U r_\mathrm{m}


    Parameters
    ----------
    phi : float
        Flow coefficient, :math:`\phi`.
    psi : float
        Stage loading coefficient, :math:`\psi`.
    Al13 : array
        Yaw angles at stage inlet and exit, :math:`(\alpha_1,\alpha_3)`.
    Ma2 : float
        Vane exit Mach number, :math:`\Ma_2`.
    ga : float
        Ratio of specific heats, :math:`\gamma`.
    eta : float
        Polytropic efficiency, :math:`\eta`.
    Vx_rat : array, default=(1.,1.)
        Axial velocity ratios, :math:`(\zeta_1,\zeta_3)`.

    Returns
    -------
    stg : NonDimStage
        Stage geometry and some secondary calculated aerodynamic parameters
        represented as a NonDimStage object.
    """

    # Rename coolant parameters for brevity
    fc = mdotc_mdot1

    #
    # First, construct velocity triangles
    #

    # Get absolute flow angles using Euler work eqn
    if psi > 0.0:
        # Turbine
        # Euler work eqn from 2-3 sets tangential velocity upstream of rotor
        tanAl2 = (
            np.tan(np.radians(Al13[1]))
            * Vx_rat[1]
            * (1.0 + fc[0] + fc[1])
            / (1.0 + fc[0])
            + psi / phi
        )
    else:
        # Compressor
        # Euler work eqn from 1-2 sets tangential velocity downstream of rotor
        tanAl2 = np.tan(np.radians(Al13[0])) * Vx_rat[0] - psi / phi

    Al2 = np.degrees(np.arctan(tanAl2))
    Al = np.insert(Al13, 1, Al2)
    cosAl = np.cos(np.radians(Al))

    # Get non-dimensional velocities from definition of flow coefficient
    Vx_U = np.array([Vx_rat[0], 1.0, Vx_rat[1]]) * phi
    Vt_U = Vx_U * np.tan(np.radians(Al))
    V_U = np.sqrt(Vx_U ** 2.0 + Vt_U ** 2.0)

    # Change reference frame for rotor-relative velocities and angles
    Vtrel_U = Vt_U - 1.0
    Vrel_U = np.sqrt(Vx_U ** 2.0 + Vtrel_U ** 2.0)
    Alrel = np.degrees(np.arctan2(Vtrel_U, Vx_U))

    # Branch depending on compressor or turbine (cooling or bleed flow)

    if psi > 0.0:

        # Vane coolant will reduce To2
        # Simple because in absolute frame
        To2_To1 = (1.0 + fc[0] * Toc_Toinf[0]) / (1.0 + fc[0])

        # Use Mach number to get U/cpTo1
        V_sqrtcpTo2 = compflow.V_cpTo_from_Ma(Ma2, ga)
        U_sqrtcpTo1 = V_sqrtcpTo2 * np.sqrt(To2_To1) / V_U[1]
        Usq_cpTo1 = U_sqrtcpTo1 ** 2.0

        # Now determine absolute stagnation temperature for rotor coolant
        rel_factor_inf = 0.5 * Usq_cpTo1 * (1.0 - 2.0 * phi * tanAl2)
        rel_factor_cool = 0.5 * Usq_cpTo1 * (1.0 - 2.0 * preswirl_factor)
        Toc2_To1 = Toc_Toinf[1] * (To2_To1 + rel_factor_inf) - rel_factor_cool
        Toc2_To2 = Toc2_To1 / To2_To1

        # Non-dimensional temperatures from U/cpTo Ma and stage loading definition
        cpTo1_Usq = 1.0 / Usq_cpTo1
        cpTo2_Usq = cpTo1_Usq * To2_To1
        cpTo3_Usq = (cpTo2_Usq * (1.0 + fc[0] + fc[1] * Toc2_To2) - psi) / (
            1.0 + fc[0] + fc[1]
        )

        # Turbine
        cpTo_Usq = np.array([cpTo1_Usq, cpTo2_Usq, cpTo3_Usq])

    else:

        # Compressor
        # Set the tip Mach number directly
        U_sqrtcpTo1 = compflow.V_cpTo_from_Ma(Ma2, ga)

        # Non-dimensional temperatures from U/cpTo Ma and stage loading definition
        cpTo1_Usq = 1.0 / U_sqrtcpTo1 ** 2
        cpTo3_Usq = cpTo1_Usq - psi

        # Compressor - constant To from 2-3
        cpTo_Usq = np.array([cpTo1_Usq, cpTo3_Usq, cpTo3_Usq])

    # Mach numbers and capacity from compressible flow relations
    Ma = compflow.Ma_from_V_cpTo(V_U / np.sqrt(cpTo_Usq), ga)
    Marel = Ma * Vrel_U / V_U
    Q = compflow.mcpTo_APo_from_Ma(Ma, ga)
    Q_Q1 = Q / Q[0]

    #
    # Second, construct annulus line
    #

    # Use polytropic effy to get entropy change
    To_To1 = cpTo_Usq / cpTo_Usq[0]
    Ds_cp = -(1.0 - 1.0 / eta) * np.log(To_To1[-1])

    # Somewhat arbitrarily, split loss using loss ratio (default 0.5)
    s_cp = np.hstack((0.0, loss_rat, 1.0)) * Ds_cp

    # Convert to stagnation pressures
    Po_Po1 = np.exp((ga / (ga - 1.0)) * (np.log(To_To1) + s_cp))

    # Account for cooling or bleed flows
    mdot_mdot1 = np.array([1.0, 1.0 + fc[0], 1 + fc[0] + fc[1]])

    # Use definition of capacity to get flow area ratios
    # Area ratios = span ratios because rm = const
    Dr_Drin = mdot_mdot1 * np.sqrt(To_To1) / Po_Po1 / Q_Q1 * cosAl[0] / cosAl

    # Evaluate some other useful secondary aerodynamic parameters
    T_To1 = To_To1 / compflow.To_T_from_Ma(Ma, ga)
    P_Po1 = Po_Po1 / compflow.Po_P_from_Ma(Ma, ga)
    Porel_Po1 = P_Po1 * compflow.Po_P_from_Ma(Marel, ga)

    # Reformulate loss as stagnation pressure loss coefficients
    # referenced to inlet dynamic head as in a compressor
    if psi > 0.0:
        # Turbine
        Lam = (T_To1[2] - T_To1[1]) / (T_To1[2] - T_To1[0])
        Yp_vane = (Po_Po1[0] - Po_Po1[1]) / (Po_Po1[0] - P_Po1[0])
        Yp_blade = (Porel_Po1[1] - Porel_Po1[2]) / (Porel_Po1[1] - P_Po1[1])
    else:
        # Compressor
        Lam = (T_To1[1] - T_To1[0]) / (T_To1[2] - T_To1[0])
        Yp_vane = (Po_Po1[1] - Po_Po1[2]) / (Po_Po1[1] - P_Po1[1])
        Yp_blade = (Porel_Po1[0] - Porel_Po1[1]) / (Porel_Po1[0] - P_Po1[0])

    # Compressor style total-to-static pressure rise coefficient
    half_rho_Usq_Po1 = (
        (0.5 * ga * Ma[0] ** 2.0)
        / compflow.Po_P_from_Ma(Ma[0], ga)
        / (V_U[0] ** 2.0)
    )
    Psi_ts = (P_Po1[2] - 1.0) / half_rho_Usq_Po1

    # Assemble all of the data into the output object
    stg = NonDimStage(
        Yp=(Yp_vane, Yp_blade),
        Al=Al,
        Alrel=Alrel,
        Ma=Ma,
        Marel=Marel,
        Ax_Ax1=Dr_Drin,
        Lam=Lam,
        U_sqrt_cpTo1=U_sqrtcpTo1,
        Po_Po1=Po_Po1,
        To_To1=To_To1,
        phi=phi,
        psi=psi,
        ga=ga,
        Vt_U=Vt_U,
        Vtrel_U=Vtrel_U,
        V_U=V_U,
        Vrel_U=Vrel_U,
        P3_Po1=P_Po1[2],
        eta=eta,
        Psi_ts=Psi_ts,
        preswirl_factor=preswirl_factor,
        mdot_mdot1=mdot_mdot1,
        fc=fc,
        TRc=Toc_Toinf,
    )

    return stg


def nondim_stage_from_Lam(
    phi,  # Flow coefficient [--]
    psi,  # Stage loading coefficient [--]
    Lam,  # Degree of reaction [--]
    Al1,  # Inlet yaw angle [deg]
    Ma2,  # Vane exit Mach number [--]
    ga,  # Ratio of specific heats [--]
    eta,  # Polytropic efficiency [--]
    Vx_rat=(1.0, 1.0),  # Axial velocity ratios [--]
    loss_rat=0.5,  # Fraction of stator loss [--]
    mdotc_mdot1=(0.0, 0.0),  # Coolant flows as fraction of inlet [--]
    Toc_Toinf=(1.0, 1.0),  # Local coolant temperature ratios [--]
    preswirl_factor=0.0,  # Preswirl factor [--]
):
    r"""Get geometry for an aerodynamic parameter set specifying reaction.

    A turbine designer is more interested in the degree of reaction of a stage,
    which controls the balance of loading between rotor and stator, rather than
    the exit yaw angle which has no such general physical interpretation.
    However, there is no analytical solution that will yield geometry at a
    fixed reaction.

    This function iterates exit yaw angle in :func:`nondim_stage_from_Al` to
    find the value which corresponds to the desired reaction, and returns a
    non-dimensional stage geometry.

    Parameters
    ----------
    phi : float
        Flow coefficient, :math:`\phi`.
    psi : float
        Stage loading coefficient, :math:`\psi`.
    Lam : float
        Degree of reaction, :math:`\Lambda`.
    Al1 : float
        Inlet yaw angle, :math:`\alpha_1`.
    Ma2 : float
        Vane exit Mach number, :math:`\Ma_2`.
    ga : float
        Ratio of specific heats, :math:`\gamma`.
    eta : float
        Polytropic efficiency, :math:`\eta`.
    Vx_rat : array, default=(1.,1.)
        Axial velocity ratios, :math:`(\zeta_1,\zeta_3)`.

    Returns
    -------
    stg : NonDimStage
        Stage geometry and some secondary calculated aerodynamic parameters
        represented as a NonDimStage object.
    """

    # Iteration step: returns error in reaction as function of exit yaw angle
    def iter_Al(x):
        stg_now = nondim_stage_from_Al(
            phi,
            psi,
            [Al1, x],
            Ma2,
            ga,
            eta,
            Vx_rat,
            loss_rat,
            mdotc_mdot1,
            Toc_Toinf,
            preswirl_factor,
        )
        return stg_now.Lam - Lam

    # Solving for Lam in general is tricky
    # Our strategy is to map out a coarse curve first, pick a point
    # close to the desired reaction, then Newton iterate

    # Evaluate guesses over entire possible yaw angle range
    Al_guess = np.linspace(-89.0, 89.0, 21)
    Lam_guess = np.zeros_like(Al_guess)

    # Catch errors if this guess of angle is horrible/non-physical
    for i in range(len(Al_guess)):
        with np.errstate(invalid="ignore"):
            try:
                Lam_guess[i] = iter_Al(Al_guess[i])
            except (ValueError, FloatingPointError):
                Lam_guess[i] = np.nan

    # Remove invalid values
    Al_guess = Al_guess[~np.isnan(Lam_guess)]
    Lam_guess = Lam_guess[~np.isnan(Lam_guess)]

    # Trim to the region between minimum and maximum reaction
    # Now the slope will be monotonic
    i1, i2 = np.argmax(Lam_guess), np.argmin(Lam_guess)
    Al_guess, Lam_guess = Al_guess[i1:i2], Lam_guess[i1:i2]

    # Start the Newton iteration at minimum error point
    i0 = np.argmin(np.abs(Lam_guess))
    # Catch the warning from scipy that derivatives are zero
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try:
            Al_soln = scipy.optimize.newton(
                iter_Al, x0=Al_guess[i0], x1=Al_guess[i0 - 1]
            )
        except:
            print("scipy warns derivatives are zero.")
            print("debug info..")
            print("Al_guess", Al_guess[i0], Al_guess[i0 - 1.0])
            print("Lam errors", iter_Al[i0 : (i0 + 2)])
            print("Al_soln", Al_soln)

    # Once we have a solution for the exit flow angle, evaluate stage geometry
    stg_out = nondim_stage_from_Al(
        phi,
        psi,
        [Al1, Al_soln],
        Ma2,
        ga,
        eta,
        Vx_rat,
        loss_rat,
        mdotc_mdot1,
        Toc_Toinf,
        preswirl_factor,
    )

    return stg_out

