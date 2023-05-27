import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

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

x1=np.array(fc1_opt)*100
y1=deta1_opt

def fit_func(x, a):
    return a * x

params = curve_fit(fit_func, x1, y1)
[a1] = params[0]

print(a1)

x2 = np.array(fc5_opt)*100
y2 = deta5_opt

params = curve_fit(fit_func, x2, y2)
[a2] = params[0]

print(a2)
plt.plot(x1, a1*x1,color='gray',label='Line of best fit')
plt.plot(x2, a2*x2,color='gray')
plt.suptitle('$\\psi$=1.78, $C_0$=0.65, $M_2$=0.67, $\\Lambda$=0.50')
plt.plot(x1,y1,label='$\\phi$=0.65',marker='x',linestyle='--',color='seagreen')
plt.plot(x2,y2,label='$\\phi$=0.95',marker='x',linestyle='--',color='darkviolet')
plt.ylabel('$\\Delta \\eta$ (%)')
plt.xlabel('$f_{\\mathrm{c,stator}} $ (%)')
plt.xlim(0)
plt.ylim(top=0.1)
plt.grid()

plt.legend()
plt.show()

x1=np.array(fc3_opt)*100
y1=deta3_opt

params = curve_fit(fit_func, x1, y1)
[a1] = params[0]

print(a1)

x2 = np.array(fc6_opt)*100
y2 = deta6_opt

params = curve_fit(fit_func, x2, y2)
[a2] = params[0]

print(a2)
plt.plot(x1, a1*x1,color='gray',label='Line of best fit')
plt.plot(x2, a2*x2,color='gray')
plt.suptitle('$\\phi$=0.81, $C_0$=0.65, $M_2$=0.67, $\\Lambda$=0.50')
plt.plot(x1,y1,label='$\\psi$=1.20',marker='x',linestyle='--',color='cornflowerblue')
plt.plot(x2,y2,label='$\\psi$=1.50',marker='x',linestyle='--',color='darkorange')
plt.ylabel('$\\Delta \\eta$ (%)')
plt.xlabel('$f_{\\mathrm{c,stator}} $ (%)')
plt.xlim(0)
plt.ylim(top=0)
plt.grid()

plt.legend()
plt.show()



