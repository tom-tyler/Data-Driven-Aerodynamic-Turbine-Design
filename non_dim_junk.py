   def nondim_to_dim(self,
                     dataframe
                     ):
      
      #   "To1": 1600.0,
      #   "Po1": 1600000.0,
      #   "rgas": 287.14,
      #   "Omega": 314.159,
      #   "delta": 0.1
      #   "htr": 0.9,

      #   "Re": 2000000.0
      
      for index, row in dataframe.iterrows():
         phi = dataframe.loc[index,'phi']                                             # Flow coefficient [--]
         psi = dataframe.loc[index,'psi']                                             # Stage loading coefficient [--]
         Al13 = (dataframe.loc[index,'Al1'],dataframe.loc[index,'Al3'])                               # Yaw angles [deg]
         Ma2 = dataframe.loc[index,'M']                                               # Vane exit Mach number [--]
         ga = 1.33                                                    # Ratio of specific heats [--]
         eta = 1.0 - dataframe.loc[index,'eta_lost']                                  # Polytropic efficiency [--]
         Vx_rat = (dataframe.loc[index,'zeta_stator'],dataframe.loc[index,'zeta_rotor'])              # Axial velocity ratios [--]
         loss_rat = dataframe.loc[index,'loss_rat']                                   # Fraction of stator loss [--]
         
         # Get absolute flow angles using Euler work eqn
         tanAl2 = (np.tan(np.radians(Al13[1])) * Vx_rat[1] + psi / phi)
         Al2 = np.degrees(np.arctan(tanAl2))
         Al = np.insert(Al13, 1, Al2)
         cosAl = np.cos(np.radians(Al))
         
         # Get non-dimensional velocities from definition of flow coefficient
         Vx_U1,Vx_U2,Vx_U3 = Vx_rat[0]*phi, phi, Vx_rat[1]*phi
         Vx_U = np.array([Vx_U1,Vx_U2,Vx_U3])
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
         
         # Assemble all of the data into the output object
         dataframe.loc[index,'Al1'],dataframe.loc[index,'Al2'],dataframe.loc[index,'Al3'] = Al  #3
         dataframe.loc[index,'Alrel1'],dataframe.loc[index,'Alrel2'],dataframe.loc[index,'Alrel3'] = Alrel  #3
         dataframe.loc[index,'M1'],dataframe.loc[index,'M2'],dataframe.loc[index,'M3'] = Ma  #3
         dataframe.loc[index,'M1rel'],dataframe.loc[index,'M2rel'],dataframe.loc[index,'M3rel'] = Marel  #3
         dataframe.loc[index,'Ax1_Ax1'],dataframe.loc[index,'Ax2_Ax1'],dataframe.loc[index,'Ax3_Ax1'] = Dr_Drin  #3
         dataframe.loc[index,'Po1_Po1'],dataframe.loc[index,'Po2_Po1'],dataframe.loc[index,'Po3_Po1'] = Po_Po1  #3
         dataframe.loc[index,'To1_To1'],dataframe.loc[index,'To2_To1'],dataframe.loc[index,'To3_To1'] = To_To1  #3
         dataframe.loc[index,'Vt1_U'],dataframe.loc[index,'Vt2_U'],dataframe.loc[index,'Vt3_U'] = Vt_U  #3
         dataframe.loc[index,'Vt1rel_U'],dataframe.loc[index,'Vt2rel_U'],dataframe.loc[index,'Vt3rel_U'] = Vtrel_U  #3
         dataframe.loc[index,'V1_U'],dataframe.loc[index,'V2_U'],dataframe.loc[index,'V3_U'] = V_U  #3
         dataframe.loc[index,'V1rel_U'],dataframe.loc[index,'V2rel_U'],dataframe.loc[index,'V3rel_U'] = Vrel_U  #3
         dataframe.loc[index,'P1_Po1'],dataframe.loc[index,'P2_Po1'],dataframe.loc[index,'P3_Po1'] = P_Po1  #3
         dataframe.loc[index,'Po1rel_Po1'],dataframe.loc[index,'Po2rel_Po1'],dataframe.loc[index,'Po3rel_Po1'] = Porel_Po1  #3
         dataframe.loc[index,'T1_To1'],dataframe.loc[index,'T2_To1'],dataframe.loc[index,'T3_To1'] = T_To1  #3
         dataframe.loc[index,'mdot1_mdot1'],dataframe.loc[index,'mdot2_mdot1'],dataframe.loc[index,'mdot3_mdot1'] = mdot_mdot1  #3

      return dataframe
   

   def nondim_to_dim_V2(self,
                     dataframe
                     ):
      
      #   "To1": 1600.0,
      #   "Po1": 1600000.0,
      #   "rgas": 287.14,
      #   "Omega": 314.159,
      #   "delta": 0.1
      #   "htr": 0.9,

      #   "Re": 2000000.0
      
      for index, row in dataframe.iterrows():
         phi = dataframe.loc[index,'phi']                                             # Flow coefficient [--]
         psi = dataframe.loc[index,'psi']                                             # Stage loading coefficient [--]
         Al13 = (dataframe.loc[index,'Al1'],dataframe.loc[index,'Al3'])                               # Yaw angles [deg]
         Ma2 = dataframe.loc[index,'M']                                               # Vane exit Mach number [--]
         ga = 1.33                                                    # Ratio of specific heats [--]
         eta = 1.0 - dataframe.loc[index,'eta_lost']                                  # Polytropic efficiency [--]
         Vx_rat = (dataframe.loc[index,'zeta_stator'],dataframe.loc[index,'zeta_rotor'])              # Axial velocity ratios [--]
         loss_rat = dataframe.loc[index,'loss_rat']                                   # Fraction of stator loss [--]
         
         # Get absolute flow angles using Euler work eqn
         tanAl2 = (np.tan(np.radians(Al13[1])) * Vx_rat[1] + psi / phi)
         Al2 = np.degrees(np.arctan(tanAl2))
         Al = np.insert(Al13, 1, Al2)
         cosAl = np.cos(np.radians(Al))
         
         # Get non-dimensional velocities from definition of flow coefficient
         Vx_U1,Vx_U2,Vx_U3 = Vx_rat[0]*phi, phi, Vx_rat[1]*phi
         Vx_U = np.array([Vx_U1,Vx_U2,Vx_U3])
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
         
         # Assemble all of the data into the output object
         dataframe.loc[index,'Al1'],dataframe.loc[index,'Al2'],dataframe.loc[index,'Al3'] = Al  #3
         dataframe.loc[index,'Alrel1'],dataframe.loc[index,'Alrel2'],dataframe.loc[index,'Alrel3'] = Alrel  #3
         dataframe.loc[index,'M1'],dataframe.loc[index,'M2'],dataframe.loc[index,'M3'] = Ma  #3
         dataframe.loc[index,'M1rel'],dataframe.loc[index,'M2rel'],dataframe.loc[index,'M3rel'] = Marel  #3
         dataframe.loc[index,'Ax1_Ax1'],dataframe.loc[index,'Ax2_Ax1'],dataframe.loc[index,'Ax3_Ax1'] = Dr_Drin  #3
         dataframe.loc[index,'Po1_Po1'],dataframe.loc[index,'Po2_Po1'],dataframe.loc[index,'Po3_Po1'] = Po_Po1  #3
         dataframe.loc[index,'To1_To1'],dataframe.loc[index,'To2_To1'],dataframe.loc[index,'To3_To1'] = To_To1  #3
         dataframe.loc[index,'Vt1_U'],dataframe.loc[index,'Vt2_U'],dataframe.loc[index,'Vt3_U'] = Vt_U  #3
         dataframe.loc[index,'Vt1rel_U'],dataframe.loc[index,'Vt2rel_U'],dataframe.loc[index,'Vt3rel_U'] = Vtrel_U  #3
         dataframe.loc[index,'V1_U'],dataframe.loc[index,'V2_U'],dataframe.loc[index,'V3_U'] = V_U  #3
         dataframe.loc[index,'V1rel_U'],dataframe.loc[index,'V2rel_U'],dataframe.loc[index,'V3rel_U'] = Vrel_U  #3
         dataframe.loc[index,'P1_Po1'],dataframe.loc[index,'P2_Po1'],dataframe.loc[index,'P3_Po1'] = P_Po1  #3
         dataframe.loc[index,'Po1rel_Po1'],dataframe.loc[index,'Po2rel_Po1'],dataframe.loc[index,'Po3rel_Po1'] = Porel_Po1  #3
         dataframe.loc[index,'T1_To1'],dataframe.loc[index,'T2_To1'],dataframe.loc[index,'T3_To1'] = T_To1  #3
         dataframe.loc[index,'mdot1_mdot1'],dataframe.loc[index,'mdot2_mdot1'],dataframe.loc[index,'mdot3_mdot1'] = mdot_mdot1  #3

      return dataframe
   