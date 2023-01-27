import numpy as np
import compflow_native as compflow
import numpy as np


def nondim_to_dim(self):
    
    phi,            # Flow coefficient [--]
    psi,            # Stage loading coefficient [--]
    Al13,           # Yaw angles [deg]
    Ma2,            # Vane exit Mach number [--]
    ga,             # Ratio of specific heats [--]
    eta,            # Polytropic efficiency [--]
    Vx_rat,         # Axial velocity ratios [--]
    loss_rat):      # Fraction of stator loss [--]

    # Get absolute flow angles using Euler work eqn
    tanAl2 = (np.tan(np.radians(Al13[1])) * Vx_rat[1] + psi / phi)
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

    # Use Mach number to get U/cpTo1
    V_sqrtcpTo2 = compflow.V_cpTo_from_Ma(Ma2, ga)
    U_sqrtcpTo1 = V_sqrtcpTo2 / V_U[1]
    Usq_cpTo1 = U_sqrtcpTo1 ** 2.0

    # Now determine absolute stagnation temperature for rotor coolant
    rel_factor_inf = 0.5 * Usq_cpTo1 * (1.0 - 2.0 * phi * tanAl2)
    rel_factor_cool = 0.5 * Usq_cpTo1
    Toc2_To1 = (1.0 + rel_factor_inf) - rel_factor_cool

    # Non-dimensional temperatures from U/cpTo Ma and stage loading definition
    cpTo1_Usq = 1.0 / Usq_cpTo1
    cpTo2_Usq = cpTo1_Usq
    cpTo3_Usq = (cpTo2_Usq - psi)

    # Turbine
    cpTo_Usq = np.array([cpTo1_Usq, cpTo2_Usq, cpTo3_Usq])
    
    # Mach numbers and capacity from compressible flow relations
    Ma = compflow.Ma_from_V_cpTo(V_U / np.sqrt(cpTo_Usq), ga)
    Marel = Ma * Vrel_U / V_U
    Q = compflow.mcpTo_APo_from_Ma(Ma, ga)
    Q_Q1 = Q / Q[0]

    # Use polytropic effy to get entropy change
    To_To1 = cpTo_Usq / cpTo_Usq[0]
    Ds_cp = -(1.0 - 1.0 / eta) * np.log(To_To1[-1])

    # Somewhat arbitrarily, split loss using loss ratio (default 0.5)
    s_cp = np.hstack((0.0, loss_rat, 1.0)) * Ds_cp

    # Convert to stagnation pressures
    Po_Po1 = np.exp((ga / (ga - 1.0)) * (np.log(To_To1) + s_cp))

    # Account for cooling or bleed flows
    mdot_mdot1 = np.array([1.0, 1.0, 1.0])

    # Use definition of capacity to get flow area ratios
    # Area ratios = span ratios because rm = const
    Dr_Drin = mdot_mdot1 * np.sqrt(To_To1) / Po_Po1 / Q_Q1 * cosAl[0] / cosAl

    # Evaluate some other useful secondary aerodynamic parameters
    T_To1 = To_To1 / compflow.To_T_from_Ma(Ma, ga)
    P_Po1 = Po_Po1 / compflow.Po_P_from_Ma(Ma, ga)
    Porel_Po1 = P_Po1 * compflow.Po_P_from_Ma(Marel, ga)
    
    # Turbine
    Lam = (T_To1[2] - T_To1[1]) / (T_To1[2] - T_To1[0])
    Yp_vane = (Po_Po1[0] - Po_Po1[1]) / (Po_Po1[0] - P_Po1[0])
    Yp_blade = (Porel_Po1[1] - Porel_Po1[2]) / (Porel_Po1[1] - P_Po1[1])
    
    # Assemble all of the data into the output object
    stg = dict([('Yp',(Yp_vane, Yp_blade)),
                ('Al',Al),
                ('Alrel',Alrel),
                ('Ma',Ma),
                ('Marel',Marel),
                ('Ax_Ax1',Dr_Drin),
                ('Lam',Lam),
                ('U_sqrt_cpTo1',U_sqrtcpTo1),
                ('Po_Po1',Po_Po1),
                ('To_To1',To_To1),
                ('phi',phi),
                ('psi',psi),
                ('ga',ga),
                ('Vt_U',Vt_U),
                ('Vtrel_U',Vtrel_U),
                ('V_U',V_U),
                ('Vrel_U',Vrel_U),
                ('P3_Po1',P_Po1[2]),
                ('eta',eta),
                ('mdot_mdot1',mdot_mdot1)])

    return stg