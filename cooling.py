from dd_turb_design import read_in_data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

factor=3
df = read_in_data('2D_tip_gap',state_retention_statistics=True,factor=factor)
lower_factor = 1 - factor/100
upper_factor = 1 + factor/100
val=0.81
df = df[df["phi"] < upper_factor*val]
df = df[df["phi"] > lower_factor*val]
val=1.78
df = df[df["psi"] < upper_factor*val]
df = df[df["psi"] > lower_factor*val]
print(df.shape)
np.mean(df['eta_lost'])

#for psi=1.78,phi=0.65
fc1_opt=[0.0,0.06]
eta_lost1_opt=np.array([0.05734,0.100100339547194])
eta_1_opt = 100-eta_lost1_opt*100
deta1_opt = eta_1_opt - eta_1_opt[0]

#for psi=1.2,phi=0.81
fc3_opt=[0.0,0.06]
eta_lost3_opt=np.array([0.05639,0.116469500287684])
eta_3_opt = 100-eta_lost3_opt*100
deta3_opt = eta_3_opt - eta_3_opt[0]

#for psi=1.78,phi=0.81

#for psi=1.78,phi=0.95
fc5_opt=[0.0,0.06]
eta_lost5_opt=np.array([0.05927,0.118396172052813])
eta_5_opt = 100-eta_lost5_opt*100
deta5_opt = eta_5_opt - eta_5_opt[0]

#for psi=1.5,phi=0.81
fc6_opt=[0.0,0.06]
eta_lost6_opt=np.array([0.054438,0.109413271578029])
eta_6_opt = 100-eta_lost6_opt*100
deta6_opt = eta_6_opt - eta_6_opt[0]

plt.suptitle('Effect of cooling [$\\psi$=1.78, $C_0$=0.65, $M_2$=0.67, $\\Lambda$=0.50]')
plt.plot(fc1_opt,deta1_opt,label='$\\phi$=0.65',marker='x',linestyle='--')
plt.plot(fc5_opt,deta5_opt,label='$\\phi$=0.95',marker='x',linestyle='--')
plt.ylabel('$\\Delta \\eta$')
plt.xlabel('$fc (stator)$')
plt.legend()
plt.xlim(0)
plt.ylim(top=0)
plt.grid()
plt.show()

plt.suptitle('Effect of cooling [$\\phi$=0.81, $C_0$=0.65, $M_2$=0.67, $\\Lambda$=0.50]')
plt.plot(fc3_opt,deta3_opt,label='$\\psi$=1.20',marker='x',linestyle='--')
plt.plot(fc6_opt,deta6_opt,label='$\\psi$=1.50',marker='x',linestyle='--')
plt.ylabel('$\\Delta \\eta$')
plt.xlabel('$fc (stator)$')
plt.legend()
plt.xlim(0)
plt.ylim(top=0)
plt.grid()
plt.show()


