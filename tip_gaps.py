import turbine_design.data_tools as tools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

factor=3
df = tools.read_in_large_dataset('2D_tip_gap',state_retention_statistics=True,factor=factor)
lower_factor = 1 - factor/100
upper_factor = 1 + factor/100
val=0.81
df = df[df["phi"] < upper_factor*val]
df = df[df["phi"] > lower_factor*val]
val=1.78
df = df[df["psi"] < upper_factor*val]
df = df[df["psi"] > lower_factor*val]
print(df.shape)
# np.mean(df['eta_lost'])

#for psi=1.78,phi=0.65
tau1=[0.0,0.035]
eta_lost1=np.array([0.05734,0.10515])
eta_1 = 100-eta_lost1*100
deta1 = eta_1 - eta_1[0]

tau1_opt=[0.0,0.005,0.02,0.05]
eta_lost1_opt=np.array([0.05734,0.0684009858038865,0.0848800187220026,0.122998337911884])
eta_1_opt = 100-eta_lost1_opt*100
deta1_opt = eta_1_opt - eta_1_opt[0]

tau1_comb=[0.0,0.005,0.02,0.035,0.05]
eta_lost1_comb=np.array([0.05734,0.0684009858038865,0.0848800187220026,0.10515,0.122998337911884])
eta_1_comb = 100-eta_lost1_comb*100
deta1_comb = eta_1_comb - eta_1_comb[0]

#for psi=1.2,phi=0.81
tau3=[0.0,0.005,0.02,0.035,0.05]
eta_lost3=np.array([0.05639,0.06367,0.075418,0.0891,0.1010])
eta_3 = 100-eta_lost3*100
deta3 = eta_3 - eta_3[0]

tau3_opt=[0.0,0.005,0.02,0.035,0.05]
eta_lost3_opt=np.array([0.05639,0.062724475615086,0.0773598359184378,0.0934432625515986,0.104747199966694])
eta_3_opt = 100-eta_lost3_opt*100
deta3_opt = eta_3_opt - eta_3_opt[0]

tau3_comb=[0.0,0.005,0.02,0.035,0.05]
eta_lost3_comb=np.mean([eta_lost3,eta_lost3_opt],axis=0)

eta_3_comb = 100-eta_lost3_comb*100
deta3_comb = eta_3_comb - eta_3_comb[0]

# for psi=1.78,phi=0.81   #v2
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

eta_lost5_optv2=np.array([0.05927,0.0879924959646095,0.107442681183047,0.117738])
tau5_comb=[0.0,0.02,0.035,0.05]
eta_lost5_comb=np.mean([eta_lost5,eta_lost5_optv2],axis=0)
eta_5_comb = 100-eta_lost5_comb*100
deta5_comb = eta_5_comb - eta_5_comb[0]

#for psi=1.5,phi=0.81
tau6=[0.0,0.02,0.035,0.05]
eta_lost6=np.array([0.054438,0.0791,0.094036,0.1092])
eta_6 = 100-eta_lost6*100
deta6 = eta_6 - eta_6[0]

tau6_opt=[0.0,0.005,0.02,0.035,0.05]
eta_lost6_opt=np.array([0.054438,0.0635177437988126,0.080058863401812,0.0974481314566691,0.110353823567772])
eta_6_opt = 100-eta_lost6_opt*100
deta6_opt = eta_6_opt - eta_6_opt[0]

eta_lost6v2=np.array([0.05927,0.0635177437988126,0.0879924959646095,0.107442681183047,0.117738])
tau6_comb=[0.0,0.005,0.02,0.035,0.05]
# eta_lost6_comb=np.mean([eta_lost6v2,eta_lost6_opt],axis=0)
eta_lost6_comb = eta_lost6v2
eta_6_comb = 100-eta_lost6_comb*100
deta6_comb = eta_6_comb - eta_6_comb[0]

def fit_func(x, a):
    return a * x

plt.suptitle('$\\psi$=1.78, $C_0$=0.65, $M_2$=0.67, $\\Lambda$=0.50')
counter = 0
for plot in [(np.array(tau1_comb)*100,deta1_comb),
             (np.array(tau4)*100,deta4),
             (np.array(tau5_comb)*100,deta5_comb)]:

    params = curve_fit(fit_func, plot[0], plot[1])
    [a] = params[0]

    # if counter == 0:
    #     plt.plot(plot[0], a*np.array(plot[0]),color='gray',label='Line of best fit')
    # else:
    #     plt.plot(plot[0], a*np.array(plot[0]),color='gray')
    # counter +=1

plt.plot(np.array(tau1_comb)*100,deta1_comb,label='$\\phi$=0.65',marker='x',color='seagreen')
plt.plot(np.array(tau4)*100,deta4,label='$\\phi$=0.81',marker='x',color='crimson')
plt.plot(np.array(tau5_comb)*100,deta5_comb,label='$\\phi$=0.95',marker='x',color='darkviolet')
# plt.plot(np.array(tau1_opt)*100,deta1_opt,marker='x',linestyle='--',color='seagreen')
# plt.plot(np.array(tau5_opt)*100,deta5_opt,marker='x',linestyle='--',color='darkviolet')
plt.ylabel('$\\Delta \\eta$ (%)')
plt.xlabel('$\\tau$ (%)')
plt.legend()
plt.xlim(0)
plt.ylim(top=0)
plt.grid()
plt.show()

plt.suptitle('$\\phi$=0.81, $C_0$=0.65, $M_2$=0.67, $\\Lambda$=0.50')
counter = 0
for plot in [(np.array(tau3_comb)*100,deta3_comb),
             (np.array(tau6)*100,deta6),
             (np.array(tau4)*100,deta4)]:
    params = curve_fit(fit_func, plot[0], plot[1])
    [a] = params[0]
    # if counter == 0:
    #     plt.plot(plot[0], a*plot[0],color='gray',label='Line of best fit')
    # else:
    #     plt.plot(plot[0], a*plot[0],color='gray')
    counter +=1

plt.plot(np.array(tau3_comb)*100,deta3_comb,label='$\\psi$=1.20',marker='x',color='cornflowerblue')
plt.plot(np.array(tau6)*100,deta6,label='$\\psi$=1.50',marker='x',color='darkorange')
plt.plot(np.array(tau4)*100,deta4,label='$\\psi$=1.78',marker='x',color='crimson')
# plt.plot(np.array(tau3_opt)*100,deta3_opt,marker='x',linestyle='--',color='cornflowerblue')
# plt.plot(np.array(tau6_opt)*100,deta6_opt,marker='x',linestyle='--',color='darkorange')
plt.ylabel('$\\Delta \\eta$ (%)')
plt.xlabel('$\\tau$ (%)')
plt.legend()
plt.xlim(0)
plt.ylim(top=0)
plt.grid()
plt.show()


