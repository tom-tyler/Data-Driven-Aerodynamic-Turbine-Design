from dd_turb_design import turbine_GPR
import numpy as np
import pandas as pd
import compflow_native as compflow
import sys


#NEED STILL
# 'recamber_te_stator',
# 'recamber_te_rotor',

# 'beta_rotor',

# 't_ps_rotor',

# 't_ss_stator',
# 't_ss_rotor',

# 'max_t_loc_ps_stator',
# 'max_t_loc_ps_rotor',

# 'max_t_loc_ss_stator',
# 'max_t_loc_ss_rotor'

# 'lean_stator',

class turbine_params:
    def __init__(self,phi,psi,M2,Co):
        
        if np.isscalar(phi) and np.isscalar(psi) and np.isscalar(M2) and np.isscalar(Co):
            self.no_points = 1
            self.phi = np.array([phi])
            self.psi = np.array([psi])
            self.M2 = np.array([M2])
            self.Co = np.array([Co])
        elif not np.isscalar(phi) and not np.isscalar(psi) and not np.isscalar(M2) and not np.isscalar(Co):
            self.phi = np.array(phi)
            self.psi = np.array(psi)
            self.M2 = np.array(M2)
            self.Co = np.array(Co)
            self.no_points = len(self.phi)
        else:
            sys.exit('Incorrect input types')
            
        self.htr = 0.9
        self.AR = [1.6,1.6]
        
        self.spf_stator,self.spf_rotor = 0.5,0.5
        self.spf = [self.spf_stator,self.spf_rotor]
        
        self.recamber_le_stator,self.recamber_le_rotor = 0.0,0.0
        self.recamber_le = [self.recamber_le_stator,self.recamber_le_rotor]
        
        self.ga = 1.33
        self.Rgas = 272.9
        self.cp = self.Rgas * self.ga / (self.ga - 1.0)
        self.tte = 0.015
        self.dx_c = [1.0,0.6,1.5]
        self.delta = 0.1
        
        #geom
        self.Rle_stator,self.Rle_rotor = [0.04,0.04]

    def get_Al(self):
    
        Al2_model = turbine_GPR('Al2a')
        Al3_model = turbine_GPR('Al3')
        
        Al1 = np.zeros(self.no_points)
        Al2 = np.array(Al2_model.predict(pd.DataFrame(data={'phi':self.phi,
                                                    'psi':self.psi,
                                                    'M2':self.M2,
                                                    'Co':self.Co}))['predicted_output'])
        Al3 = np.array(Al3_model.predict(pd.DataFrame(data={'phi':self.phi,
                                                    'psi':self.psi,
                                                    'M2':self.M2,
                                                    'Co':self.Co}))['predicted_output'])
        self.Al1 = Al1
        self.Al2 = Al2
        self.Al3 = Al3
        self.Al = np.array([Al1,Al2,Al3])
        
        return self.Al

    def get_stagger(self):
        
        stagger_stator_model = turbine_GPR('stagger_stator')
        stagger_rotor_model = turbine_GPR('stagger_rotor')
        
        stagger_stator = np.array(stagger_stator_model.predict(pd.DataFrame(data={'phi':self.phi,
                                                                        'psi':self.psi,
                                                                        'M2':self.M2,
                                                                        'Co':self.Co}))['predicted_output'])
        stagger_rotor = np.array(stagger_rotor_model.predict(pd.DataFrame(data={'phi':self.phi,
                                                                        'psi':self.psi,
                                                                        'M2':self.M2,
                                                                        'Co':self.Co}))['predicted_output'])
        self.stagger_stator = stagger_stator
        self.stagger_rotor = stagger_rotor
        self.stagger = np.array([stagger_stator,stagger_rotor])
        return self.stagger

    def get_zeta(self):
        zeta_stator_model = turbine_GPR('zeta_stator') #maybe improve this model
        
        zeta_stator = np.array(zeta_stator_model.predict(pd.DataFrame(data={'phi':self.phi,
                                                    'psi':self.psi,
                                                    'M2':self.M2,
                                                    'Co':self.Co}))['predicted_output'])
        zeta_rotor = np.ones(self.no_points)
        
        self.zeta_stator = zeta_stator
        self.zeta_rotor = zeta_rotor
        self.zeta = np.array([zeta_stator,zeta_rotor])
        
        return self.zeta

    def get_s_cx(self):
        s_cx_stator_model = turbine_GPR('s_cx_stator')
        s_cx_rotor_model = turbine_GPR('s_cx_rotor')
        
        s_cx_stator = np.array(s_cx_stator_model.predict(pd.DataFrame(data={'phi':self.phi,
                                                    'psi':self.psi,
                                                    'M2':self.M2,
                                                    'Co':self.Co}))['predicted_output'])
        s_cx_rotor = np.array(s_cx_rotor_model.predict(pd.DataFrame(data={'phi':self.phi,
                                                    'psi':self.psi,
                                                    'M2':self.M2,
                                                    'Co':self.Co}))['predicted_output'])
        
        self.s_cx_stator = s_cx_stator
        self.s_cx_rotor = s_cx_rotor
        self.s_cx = np.array([s_cx_stator,s_cx_rotor])
        
        return self.s_cx
    
    def get_loss_rat(self):
        loss_rat_model = turbine_GPR('loss_rat')
        
        self.get_Yp()
        
        self.loss_rat = np.array(loss_rat_model.predict(pd.DataFrame(data={'phi':self.phi,
                                                    'psi':self.psi,
                                                    'Yp_stator':self.Yp_stator,
                                                    'Yp_rotor':self.Yp_rotor,
                                                    'Co':self.Co}))['predicted_output'])
        
        return self.loss_rat

    def get_eta_lost(self):
        eta_lost_model = turbine_GPR('eta_lost')
        
        self.get_Yp()
        
        self.eta_lost = np.array(eta_lost_model.predict(pd.DataFrame(data={'phi':self.phi,
                                                    'psi':self.psi,
                                                    'M2':self.M2,
                                                    'Co':self.Co,
                                                    'Yp_stator':self.Yp_stator,
                                                    'Yp_rotor':self.Yp_rotor}))['predicted_output'])
        
        return self.eta_lost
    
    def get_t_ps(self):
        self.t_ps_stator = 0.205*np.ones(self.no_points)
        self.t_ps_rotor = 0.250*np.ones(self.no_points)
        self.t_ps = np.array([self.t_ps_stator,self.t_ps_rotor])
        return self.t_ps
    
    def get_t_ss(self):
        self.t_ss_stator = 0.29*np.ones(self.no_points)
        self.t_ss_rotor = 0.30*np.ones(self.no_points)
        self.t_ss = np.array([self.t_ss_stator,self.t_ps_rotor])
        return self.t_ss

    def get_Yp(self):
        Yp_stator_model = turbine_GPR('Yp_stator')
        Yp_rotor_model = turbine_GPR('Yp_rotor')
        
        self.get_stagger()
        self.get_s_cx()
        self.get_Al()
        
        Yp_stator = np.array(Yp_stator_model.predict(pd.DataFrame(data={'s_cx_stator':self.s_cx_stator,
                                                                        'stagger_stator':self.stagger_stator,
                                                                        'M2':self.M2,
                                                                        'Al2a':self.Al2}))['predicted_output'])
        Yp_rotor = np.array(Yp_rotor_model.predict(pd.DataFrame(data={'s_cx_rotor':self.s_cx_rotor,
                                                                        'psi':self.psi,
                                                                        'M2':self.M2,
                                                                        'stagger_rotor':self.stagger_rotor}))['predicted_output'])
        self.Yp_stator = Yp_stator
        self.Yp_rotor = Yp_rotor
        self.Yp = np.array([Yp_stator,Yp_rotor])
        return self.Yp
    
    def get_beta(self):
        self.beta_stator = 10.5*np.ones(self.no_points)
        return self.beta
    
    def get_lean(self):
        self.lean_stator = 0.03*np.ones(self.no_points)  #ballpark
        self.lean_rotor = np.zeros(self.no_points)
        self.lean = np.array([self.lean_stator,self.lean_rotor])
        return self.lean

    def get_recamber_te(self):
        self.recamber_te_stator = np.zeros(self.no_points) #ballpark
        self.recamber_te_rotor = np.zeros(self.no_points)  #ballpark
        self.recamber_te = np.array([self.recamber_te_stator,self.recamber_te_rotor])
        return self.recamber_te
    
    def get_max_t_loc_ps(self):
        self.max_t_loc_ps_stator = 0.35*np.ones(self.no_points)   #ballpark
        self.max_t_loc_ps_rotor = 0.37*np.ones(self.no_points)    #ballpark
        self.max_t_loc_ps = np.array([self.max_t_loc_ps_stator,self.max_t_loc_ps_rotor])
        return self.max_t_loc_ps
    
    def get_max_t_loc_ss(self):
        self.max_t_loc_ss_stator = 0.40*np.ones(self.no_points)   #ballpark
        self.max_t_loc_ss_rotor = 0.32*np.ones(self.no_points)    #ballpark
        self.max_t_loc_ss = np.array([self.max_t_loc_ss_stator,self.max_t_loc_ps_rotor])
        return self.max_t_loc_ss
    
    def non_dim_params_from_4D(self):

        Al = np.array(self.get_Al())
        loss_ratio = self.get_lost_rat()
        eta_lost = self.get_eta_lost()
        
        zeta = self.get_zeta() #zeta rotor assumed=1.0
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
        
        Al = np.array(self.get_Al())
        
        rh_vane = rh[:2].reshape(-1, 1)
        rc_vane = rc[:2].reshape(-1, 1)
        Al_vane = Al[:2].reshape(-1, 1)

        r_rm = (np.reshape(self.spf_stator, (1, -1)) * (rh_vane - rc_vane) + rh_vane) / rm

        return np.degrees(np.arctan(np.tan(np.radians(Al_vane)) / r_rm))

    def free_vortex_blade(self,rh,rc,rm):
        """Evaluate blade flow angles assuming a free vortex."""
        
        Al = np.array(self.get_Al())
        
        rh_blade = rh[1:].reshape(-1, 1)
        rc_blade = rc[1:].reshape(-1, 1)
        Al_blade = Al[1:].reshape(-1, 1)

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
        
        s_cx = self.get_s_cx()
        
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
        
        self.stag1 = np.mean(self.chi,axis=1)
        self.stag2 = self.get_stagger()
        
        return self.stag1, self.stag2
 
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
        
        s_cx = self.get_s_cx()
        
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
        self.stag1 = np.mean(self.chi,axis=1)
        self.stag2 = self.get_stagger()
        
        return self.stag1, self.stag2

    def get_non_dim_geometry(self):
        self.get_stagger()
        self.get_s_cx()
        self.get_t_ps()
        self.get_t_ss()
        self.get_max_t_loc_ps()
        self.get_max_t_loc_ss()
        self.get_recamber_te()
        self.get_lean()
        self.get_beta()
        
        sect_row_0_dict = {'tte':self.tte,
                           'sect_0': {
                               'spf':self.spf_stator,
                               'stagger':self.stagger_stator,
                               'recamber':[self.recamber_le_stator,
                                           self.recamber_te_stator],
                               'Rle':self.Rle_stator,
                               'beta':self.beta_stator,
                               "thickness_ps": self.t_ps_stator,
                               "thickness_ss": self.t_ss_stator,
                               "max_thickness_location_ss": self.max_loc_t_ss_stator,
                               "max_thickness_location_ps": self.max_loc_t_ps_stator,
                               "lean": self.lean_stator
                               }
                           }
        
        sect_row_1_dict = {'tte':self.tte,
                           'sect_0': {
                               'spf':self.spf_rotor,
                               'stagger':self.stagger_rotor,
                               'recamber':[self.recamber_le_rotor,
                                           self.recamber_te_rotor],
                               'Rle':self.Rle_rotor,
                               'beta':self.beta_rotor,
                               "thickness_ps": self.t_ps_rotor,
                               "thickness_ss": self.t_ss_rotor,
                               "max_thickness_location_ss": self.max_loc_t_ss_rotor,
                               "max_thickness_location_ps": self.max_loc_t_ps_rotor,
                               "lean": self.lean_rotor
                               }
                           }
        
        return [sect_row_0_dict,sect_row_1_dict]
        
        