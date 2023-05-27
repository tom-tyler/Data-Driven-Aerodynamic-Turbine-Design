from turbine_design import turbine_design as TD
from turbine_design import data_tools as tools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcol

fig,axes = plt.subplots(1,3,sharey=True,squeeze=True)

n = 250
PHI,PSI = np.meshgrid(np.linspace(0.5,1.1,n),np.linspace(1.2,2.2,n))
# phi_vector = np.linspace(0.5,1.1,n)
# psi_vector = np.linspace(1.2,2.2,n)
# phi_vector = PHI.ravel()
# psi_vector = PSI.ravel()

# pitch_to_chord = turb.s_cx_stator.reshape(n,n)
# eta_lost = turb.eta_lost.reshape(n,n)

M_array = [0.55,0.7,0.85]

for mi,ax in enumerate(axes):

    opt_pitch_to_chord = np.zeros([n,n])

    # for i in range(n*n):
    for i in range(n):
        print(f'[{i}/{n}]')
        for j in range(n):
            k=10
            phi=PHI[i][j]*np.ones(k)
            psi=PSI[i][j]*np.ones(k)
            M2=M_array[mi]*np.ones(k)
            Co=np.linspace(0.4,0.8,k)

            turb = TD.turbine(phi,
                            psi,
                            M2,
                            Co)

            pitch_to_chord = turb.s_cx_stator
            eta_lost = turb.eta_lost
            
            min_eta_lost = np.amin(eta_lost)
            min_i = np.where(eta_lost == min_eta_lost)
            opt_pitch_to_chord[i][j] = pitch_to_chord[min_i]


            # plt.plot(pitch_to_chord,eta_lost,label=f'$\\psi$={psi_val}, $\\phi$={phi_val}')

    # opt_pitch_to_chord=opt_pitch_to_chord.reshape(n,n)
    contour_levels=[0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7]
    color_limits = np.array([0.8,1.3,1.8])
    cmap_colors = ["blue","purple","orange"]
    cmap_norm=plt.Normalize(min(color_limits),max(color_limits))
    cmap_tuples = list(zip(map(cmap_norm,color_limits), cmap_colors))
    output_cmap = mcol.LinearSegmentedColormap.from_list("", cmap_tuples)
    cplot = ax.contour(PHI,PSI,opt_pitch_to_chord,
                        levels=contour_levels,cmap=output_cmap,norm=cmap_norm)
    ax.clabel(cplot, inline=1, fontsize=12)
    ax.set_xlabel('$\\phi$',size=12)
    ax.set_ylabel('$\\psi$',size=12)
    ax.grid(linestyle = '--', linewidth = 0.5)
    ax.set_title(f'$M_2={M_array[mi]:.2f}$',size=12)
# plt.title('s_cx')
# plt.legend()
plt.show()