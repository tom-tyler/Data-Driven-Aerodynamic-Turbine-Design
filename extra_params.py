from dd_turb_design import turbine_GPR
import numpy as np
import pandas as pd
import compflow_native as compflow

#specify 2 of:
# - mass flow rate
# - shaft speed, omega
# - inlet enthalpy
# - mean radius

class turbine_info_4D:
    def __init__(self,phi,psi,M2,Co):
        self.phi = phi
        self.psi = psi
        self.M2 = M2
        self.Co = Co
        self.htr = 0.9
        self.AR = [1.6,1.6]
        self.spf_stator,self.spf_rotor = 0.5,0.5
        self.recamber_le_stator,self.recamber_le_rotor = 0.0,0.0
        self.lean_rotor = 0.0
        self.ga = 1.33
        self.Rgas = 272.9
        self.cp = self.Rgas * self.ga / (self.ga - 1.0)
        self.tte = 0.015
        self.dx_c = [1.0,0.6,1.5]
        delta = 0.1

    def Al_from_4D(self):
    
        Al2_model = turbine_GPR('Al2a')
        Al3_model = turbine_GPR('Al3')
        
        Al1 = 0.0
        Al2 = float(Al2_model.predict(pd.DataFrame(data={'phi':[self.phi],
                                                    'psi':[self.psi],
                                                    'M2':[self.M2],
                                                    'Co':[self.Co]}))['predicted_output'])
        Al3 = float(Al3_model.predict(pd.DataFrame(data={'phi':[self.phi],
                                                    'psi':[self.psi],
                                                    'M2':[self.M2],
                                                    'Co':[self.Co]}))['predicted_output'])
        
        return Al1,Al2,Al3

    def stagger_from_4D(self):
        stagger_stator_model = turbine_GPR('stagger_stator')
        stagger_rotor_model = turbine_GPR('stagger_rotor')
        
        stagger_stator = float(stagger_stator_model.predict(pd.DataFrame(data={'phi':[self.phi],
                                                    'psi':[self.psi],
                                                    'M2':[self.M2],
                                                    'Co':[self.Co]}))['predicted_output'])
        stagger_rotor = float(stagger_rotor_model.predict(pd.DataFrame(data={'phi':[self.phi],
                                                    'psi':[self.psi],
                                                    'M2':[self.M2],
                                                    'Co':[self.Co]}))['predicted_output'])
        
        return [stagger_stator,stagger_rotor]

    def zeta_from_4D(self):
        zeta_stator_model = turbine_GPR('zeta_stator')
        
        zeta_stator = float(zeta_stator_model.predict(pd.DataFrame(data={'phi':[self.phi],
                                                    'psi':[self.psi],
                                                    'M2':[self.M2],
                                                    'Co':[self.Co]}))['predicted_output'])
        zeta_rotor = 1.0
        
        return [zeta_stator,zeta_rotor]

    def s_cx_from_4D(self):
        s_cx_stator_model = turbine_GPR('s_cx_stator')
        s_cx_rotor_model = turbine_GPR('s_cx_rotor')
        
        s_cx_stator = float(s_cx_stator_model.predict(pd.DataFrame(data={'phi':[self.phi],
                                                    'psi':[self.psi],
                                                    'M2':[self.M2],
                                                    'Co':[self.Co]}))['predicted_output'])
        s_cx_rotor = float(s_cx_rotor_model.predict(pd.DataFrame(data={'phi':[self.phi],
                                                    'psi':[self.psi],
                                                    'M2':[self.M2],
                                                    'Co':[self.Co]}))['predicted_output'])
        
        return [s_cx_stator,s_cx_rotor]
    
    def loss_rat_from_4D(self):
        loss_rat_model = turbine_GPR('loss_rat')
        
        loss_rat = float(loss_rat_model.predict(pd.DataFrame(data={'phi':[self.phi],
                                                    'psi':[self.psi],
                                                    'M2':[self.M2],
                                                    'Co':[self.Co]}))['predicted_output'])
        
        return loss_rat

    def eta_lost_from_4D(self):
        eta_lost_model = turbine_GPR('eta_lost')
        
        eta_lost = float(eta_lost_model.predict(pd.DataFrame(data={'phi':[self.phi],
                                                    'psi':[self.psi],
                                                    'M2':[self.M2],
                                                    'Co':[self.Co]}))['predicted_output'])
        
        return eta_lost

    def non_dim_params_from_4D(self):
                       
        Al1,Al2,Al3 = self.Al_from_4D()
        Al = np.array([Al1,Al2,Al3])
        loss_ratio = self.loss_rat_from_4D()
        eta_lost = self.eta_lost_from_4D()
        
        zeta = self.zeta_from_4D() #zeta rotor assumed=1.0
        cosAl = np.cos(np.radians(Al))
            
        # Get non-dimensional velocities from definition of flow coefficient
        Vx_U1,Vx_U2,Vx_U3 = self.phi*zeta[0], self.phi, self.phi*zeta[1]
        Vx_U = np.array([Vx_U1,Vx_U2,Vx_U3])
        Vt_U = Vx_U * np.tan(np.radians(Al))
        V_U = np.sqrt(Vx_U ** 2.0 + Vt_U ** 2.0)

        # Change reference frame for rotor-relative velocities and angles
        Vtrel_U = Vt_U - 1.0
        Vrel_U = np.sqrt(Vx_U ** 2.0 + Vtrel_U ** 2.0)
        Alrel = np.degrees(np.arctan2(Vtrel_U, Vx_U))

        # Use Mach number to get U/cpTo1
        V_sqrtcpTo2 = compflow.V_cpTo_from_Ma(self.M2, self.ga)
        U_sqrtcpTo1 = V_sqrtcpTo2 / V_U[1]
        
        Usq_cpTo1 = U_sqrtcpTo1 ** 2.0

        # Non-dimensional temperatures from U/cpTo Ma and stage loading definition
        cpTo1_Usq = 1.0 / Usq_cpTo1
        cpTo2_Usq = cpTo1_Usq
        cpTo3_Usq = (cpTo2_Usq - self.psi)

        # Turbine
        cpTo_Usq = np.array([cpTo1_Usq, cpTo2_Usq, cpTo3_Usq])
        
        # Mach numbers and capacity from compressible flow relations
        Ma = compflow.Ma_from_V_cpTo(V_U / np.sqrt(cpTo_Usq), self.ga)
        Marel = Ma * Vrel_U / V_U
        Q = compflow.mcpTo_APo_from_Ma(Ma, self.ga)
        Q_Q1 = Q / Q[0]

        # Use polytropic effy to get entropy change
        To_To1 = cpTo_Usq / cpTo_Usq[0]
        Ds_cp = -(1.0 - 1.0 / (1.0 - eta_lost)) * np.log(To_To1[-1])

        # Somewhat arbitrarily, split loss using loss ratio (default 0.5)
        s_cp = np.hstack((0.0, loss_ratio, 1.0)) * Ds_cp

        # Convert to stagnation pressures
        Po_Po1 = np.exp((self.ga / (self.ga - 1.0)) * (np.log(To_To1) + s_cp))

        # Account for cooling or bleed flows
        mdot_mdot1 = np.array([1.0, 1.0, 1.0])

        # Use definition of capacity to get flow area ratios
        # Area ratios = span ratios because rm = const
        Dr_Drin = mdot_mdot1 * np.sqrt(To_To1) / Po_Po1 / Q_Q1 * cosAl[0] / cosAl

        # Evaluate some other useful secondary aerodynamic parameters
        T_To1 = To_To1 / compflow.To_T_from_Ma(Ma, self.ga)
        P_Po1 = Po_Po1 / compflow.Po_P_from_Ma(Ma, self.ga)
        Porel_Po1 = P_Po1 * compflow.Po_P_from_Ma(Marel, self.ga)
        
        # Turbine
        Lam = (T_To1[2] - T_To1[1]) / (T_To1[2] - T_To1[0])
        
        self.Al = Al
        self.Alrel = Alrel
        self.Ma = Ma
        self.Marel =Marel
        self.Ax_Ax1 = Dr_Drin
        self.U_sqrtcpTo1 = U_sqrtcpTo1
        self.Po_Po1 = Po_Po1
        self.To_To1 = To_To1
        self.Vt_U = Vt_U
        self.Vtrel_U = Vtrel_U
        self.V_U = V_U
        self.Vrel_U = Vrel_U
        self.P_Po1 = P_Po1
        self.Porel_Po1 = Porel_Po1
        self.T_To1 = T_To1
        self.mdot_mdot1 = mdot_mdot1
        self.Lam = Lam

    def free_vortex_vane(self,rh,rc,rm):
        """Evaluate vane flow angles assuming a free vortex."""

        rh_vane = rh[:2].reshape(-1, 1)
        rc_vane = rc[:2].reshape(-1, 1)
        Al_vane = self.Al[:2].reshape(-1, 1)

        r_rm = (np.reshape(self.spf_stator, (1, -1)) * (rh_vane - rc_vane) + rh_vane) / rm

        return np.degrees(np.arctan(np.tan(np.radians(Al_vane)) / r_rm))

    def free_vortex_blade(self,rh,rc,rm):
        """Evaluate blade flow angles assuming a free vortex."""

        rh_blade = rh[1:].reshape(-1, 1)
        rc_blade = rc[1:].reshape(-1, 1)
        Al_blade = self.Al[1:].reshape(-1, 1)

        r_rm = (np.reshape(self.spf_rotor, (1, -1)) * (rc_blade - rh_blade) + rh_blade) / rm

        return np.degrees(np.arctan(np.tan(np.radians(Al_blade)) / r_rm - r_rm / self.phi))

    def dim_from_omega(self, Omega, To1, Po1):
        """Scale a mean-line design and evaluate geometry from omega."""
        
        self.non_dim_params_from_4D()

        cpTo1 = self.cp * To1
        U = self.U_sqrtcpTo1 * np.sqrt(cpTo1)
        rm = U / Omega

        # Use hub-to-tip ratio to set span (mdot will therefore float)
        Dr_rm = 2.0 * (1.0 - self.htr) / (1.0 + self.htr)
        Dr = rm * Dr_rm * np.array(self.Ax_Ax1) / self.Ax_Ax1[1]

        Q1 = compflow.mcpTo_APo_from_Ma(self.Ma[0], self.ga)

        Ax1 = 2.0 * np.pi * rm * Dr[0]
        mdot1 = Q1 * Po1 * Ax1 * np.cos(np.radians(self.Al[0])) / np.sqrt(cpTo1)

        # Chord from aspect ratio
        span = np.array([np.mean(Dr[i : (i + 2)]) for i in range(2)])
        cx = span / self.AR
        
        s_cx = self.s_cx_from_4D()
        
        self.rm = rm
        self.U = U
        self.Dr = Dr
        self.rh = rm - Dr / 2.0
        self.rc = rm + Dr / 2.0
        self.Ax1 = Ax1
        self.mdot1 = mdot1
        
        self.span = span
        self.chord_x = cx
        self.pitch_stator = s_cx[0]*cx
        self.pitch_rotor = s_cx[1]*cx
        
        self.Omega = Omega
        self.Po1 = Po1
        self.To1 = To1
        
        self.chi = np.stack((self.free_vortex_vane(self.rh,self.rc,self.rm),
                             self.free_vortex_blade(self.rh,self.rc,self.rm)
                             ))
 
    def dim_from_mdot(self, mdot1, To1, Po1):
        """Scale a mean-line design and evaluate geometry from mdot."""
        
        self.non_dim_params_from_4D()

        cpTo1 = self.cp * To1
        Q1 = compflow.mcpTo_APo_from_Ma(self.Ma[0], self.ga)
        
        Ax1 = np.sqrt(cpTo1) * mdot1 / (Q1 * Po1 * np.cos(np.radians(self.Al[0])))
        Dr_rm = 2.0 * (1.0 - self.htr) / (1.0 + self.htr)
        
        rm = np.sqrt(Ax1 * self.Ax_Ax1[1] / (2.0 * np.pi * Dr_rm * self.Ax_Ax1[0])) 
        
        U = self.U_sqrtcpTo1 * np.sqrt(cpTo1)
        Omega = U / rm

        Dr = rm * Dr_rm * np.array(self.Ax_Ax1) / self.Ax_Ax1[1]

        # Chord from aspect ratio
        span = np.array([np.mean(Dr[i : (i + 2)]) for i in range(2)])
        cx = span / self.AR
        
        s_cx = self.s_cx_from_4D()
        
        self.rm = rm
        self.U = U
        self.Dr = Dr
        self.rh = rm - Dr / 2.0
        self.rc = rm + Dr / 2.0
        self.Ax1 = Ax1
        self.mdot1 = mdot1
        
        self.span = span
        self.chord_x = cx
        self.pitch_stator = s_cx[0]*cx
        self.pitch_rotor = s_cx[1]*cx
        
        self.Omega = Omega
        self.Po1 = Po1
        self.To1 = To1
        
        self.chi = np.stack((self.free_vortex_vane(self.rh,self.rc,self.rm),
                             self.free_vortex_blade(self.rh,self.rc,self.rm)
                             ))
        
        return self.chi

