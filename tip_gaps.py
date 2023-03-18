from turbine_design.data_tools import read_in_data
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
tau1=[0.0,0.035]
eta_lost1=np.array([0.05734,0.10515])
eta_1 = 100-eta_lost1*100
deta1 = eta_1 - eta_1[0]

tau1_opt=[0.0,0.005,0.02,0.05]
eta_lost1_opt=np.array([0.05734,0.0684009858038865,0.0848800187220026,0.122998337911884])
eta_1_opt = 100-eta_lost1_opt*100
deta1_opt = eta_1_opt - eta_1_opt[0]

#for psi=1.2,phi=0.81
tau3=[0.0,0.005,0.02,0.035,0.05]
eta_lost3=np.array([0.05639,0.06367,0.075418,0.0891,0.1010])
eta_3 = 100-eta_lost3*100
deta3 = eta_3 - eta_3[0]

tau3_opt=[0.0,0.005,0.02,0.035,0.05]
eta_lost3_opt=np.array([0.05639,0.062724475615086,0.0773598359184378,0.0934432625515986,0.104747199966694])
eta_3_opt = 100-eta_lost3_opt*100
deta3_opt = eta_3_opt - eta_3_opt[0]

#for psi=1.78,phi=0.81   #v2
tau4=[0.0,0.005,0.02,0.035,0.05]
eta_lost4=np.array([np.mean(df['eta_lost']),0.06496,0.08324,0.10248,0.11728])
eta_4 = 100-eta_lost4*100
deta4 = eta_4 - eta_4[0]

#for psi=1.78,phi=0.95
tau5=[0.0,0.02,0.035,0.05]
eta_lost5=np.array([0.05927,0.0860,0.102859,0.117738])
eta_5 = 100-eta_lost5*100
deta5 = eta_5 - eta_5[0]

tau5_opt=[0.0,0.02,0.035]
eta_lost5_opt=np.array([0.05927,0.0879924959646095,0.107442681183047])
eta_5_opt = 100-eta_lost5_opt*100
deta5_opt = eta_5_opt - eta_5_opt[0]

#for psi=1.5,phi=0.81
tau6=[0.0,0.02,0.035,0.05]
eta_lost6=np.array([0.054438,0.0791,0.094036,0.1092])
eta_6 = 100-eta_lost6*100
deta6 = eta_6 - eta_6[0]

tau6_opt=[0.0,0.005,0.02,0.035,0.05]
eta_lost6_opt=np.array([0.054438,0.0635177437988126,0.080058863401812,0.0974481314566691,0.110353823567772])
eta_6_opt = 100-eta_lost6_opt*100
deta6_opt = eta_6_opt - eta_6_opt[0]

plt.suptitle('Effect of tip gaps [$\\psi$=1.78, $C_0$=0.65, $M_2$=0.67, $\\Lambda$=0.50]')
plt.plot(tau1,deta1,label='$\\phi$=0.65',marker='x')
plt.plot(tau4,deta4,label='$\\phi$=0.81',marker='x')
plt.plot(tau5,deta5,label='$\\phi$=0.95',marker='x')
plt.plot(tau1_opt,deta1_opt,label='$\\phi$=0.65',marker='x',linestyle='--')
plt.plot(tau5_opt,deta5_opt,label='$\\phi$=0.95',marker='x',linestyle='--')
plt.ylabel('$\\Delta \\eta$')
plt.xlabel('$\\tau$')
plt.legend()
plt.xlim(0)
plt.ylim(top=0)
plt.grid()
plt.show()

plt.suptitle('Effect of tip gaps [$\\phi$=0.81, $C_0$=0.65, $M_2$=0.67, $\\Lambda$=0.50]')
plt.plot(tau3,deta3,label='$\\psi$=1.20',marker='x')
plt.plot(tau6,deta6,label='$\\psi$=1.50',marker='x')
plt.plot(tau4,deta4,label='$\\psi$=1.78',marker='x')
plt.plot(tau3_opt,deta3_opt,label='$\\psi$=1.20',marker='x',linestyle='--')
plt.plot(tau6_opt,deta6_opt,label='$\\psi$=1.50',marker='x',linestyle='--')
plt.ylabel('$\\Delta \\eta$')
plt.xlabel('$\\tau$')
plt.legend()
plt.xlim(0)
plt.ylim(top=0)
plt.grid()
plt.show()


