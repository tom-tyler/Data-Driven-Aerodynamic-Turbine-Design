# from turbine_design.data_tools import read_in_data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# factor=3
# df = read_in_data('2D_tip_gap',state_retention_statistics=True,factor=factor)
# lower_factor = 1 - factor/100
# upper_factor = 1 + factor/100
# val=0.81
# df = df[df["phi"] < upper_factor*val]
# df = df[df["phi"] > lower_factor*val]
# val=1.78
# df = df[df["psi"] < upper_factor*val]
# df = df[df["psi"] > lower_factor*val]
# print(df.shape)
# np.mean(df['eta_lost'])

#for psi=1.78,phi=0.65
fc1_opt=[0.0,0.015,0.030,0.045,0.060,0.075]
eta_lost1_opt=np.array([0.05734,
                        0.0571912964566352,
                        0.0753198835462712,
                        0.0919604037463436,
                        0.100100339547194,
                        0.107932280368586])
eta_1_opt = 100-eta_lost1_opt*100
deta1_opt = eta_1_opt - eta_1_opt[0]

#for psi=1.2,phi=0.81
fc3_opt=[0.0,0.015,0.030,0.045,0.060]
eta_lost3_opt=np.array([0.05639,
                        0.0676163933708763,
                        0.0907923534408338,
                        0.105104815865362,
                        0.116469500287684])
eta_3_opt = 100-eta_lost3_opt*100
deta3_opt = eta_3_opt - eta_3_opt[0]

#for psi=1.78,phi=0.81

#for psi=1.78,phi=0.95
fc5_opt=[0.0,0.015,0.030,0.045,0.060,0.075]
eta_lost5_opt=np.array([0.05927,
                        0.0688114751678964,
                        0.0949298675515177,
                        0.108688114231606,
                        0.118396172052813,
                        0.127909634541184])
eta_5_opt = 100-eta_lost5_opt*100
deta5_opt = eta_5_opt - eta_5_opt[0]

#for psi=1.5,phi=0.81
fc6_opt=[0.0,0.015,0.030,0.045,0.060,0.075]
eta_lost6_opt=np.array([0.054438,
                        0.0624979828942724,
                        0.0870888195710426,
                        0.0931916659185694,
                        0.109413271578029,
                        0.117604559827089])
eta_6_opt = 100-eta_lost6_opt*100
deta6_opt = eta_6_opt - eta_6_opt[0]

plt.suptitle('$\\psi$=1.78, $C_0$=0.65, $M_2$=0.67, $\\Lambda$=0.50')
plt.plot(np.array(fc1_opt)*100,deta1_opt,label='$\\phi$=0.65',marker='x',linestyle='--',color='seagreen')
plt.plot(np.array(fc5_opt)*100,deta5_opt,label='$\\phi$=0.95',marker='x',linestyle='--',color='darkviolet')
plt.ylabel('$\\Delta \\eta$ (%)')
plt.xlabel('$f_{\\mathrm{c,stator}} $ (%)')
plt.legend()
plt.xlim(0)
plt.ylim(top=0.1)
plt.grid()
plt.show()

plt.suptitle('$\\phi$=0.81, $C_0$=0.65, $M_2$=0.67, $\\Lambda$=0.50')
plt.plot(np.array(fc3_opt)*100,deta3_opt,label='$\\psi$=1.20',marker='x',linestyle='--',color='cornflowerblue')
plt.plot(np.array(fc6_opt)*100,deta6_opt,label='$\\psi$=1.50',marker='x',linestyle='--',color='darkorange')
plt.ylabel('$\\Delta \\eta$ (%)')
plt.xlabel('$f_{\\mathrm{c,stator}} $ (%)')
plt.legend()
plt.xlim(0)
plt.ylim(top=0)
plt.grid()
plt.show()


