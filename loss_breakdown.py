from turbine_design import turbine_GPR, turbine, read_in_data, split_data, read_in_large_dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcol

# data = read_in_large_dataset('4D')
# datatr,datate =split_data(data)

# Yp_stator_model = turbine_GPR('Yp_stator_model_phi_psi_M2_Co')
# Yp_rotor_model = turbine_GPR('Yp_rotor_model_phi_psi_M2_Co')

# loss_rat_model = turbine_GPR('loss_rat_model_phi_psi_M2_Co')

# loss_rat_model.plot_accuracy(datate)

n = 100
X1,X2 = np.meshgrid(np.linspace(0.4,1.2,n),np.linspace(1.0,2.4,n))
X1_vector = X1.ravel() #vector of "all" x coordinates from meshgrid
X2_vector = X2.ravel() #vector of "all" y coordinates from meshgrid
         
fig,axes = plt.subplots(1,3,sharey=True,squeeze=True)

M_array = [0.55,0.70,0.85]

for mi,ax in enumerate(axes):
    turb = turbine(X1_vector,
                    X2_vector,
                    M_array[mi]*np.ones(len(X1_vector)),
                    0.65*np.ones(len(X1_vector)))

    loss_rat = turb.loss_rat.reshape(n,n)
    Yp_s = turb.Yp[0].reshape(n,n)
    Yp_r = turb.Yp[1].reshape(n,n)


    # Yp_frac = Yp_r#/(Yp_s+Yp_r)
    Yp_frac = 1 - loss_rat
    
    # opt_pitch_to_chord=opt_pitch_to_chord.reshape(n,n)
    contour_levels=np.arange(0.5,0.74,0.02)
    color_limits = np.array([0.5,0.6,0.72])
    # contour_levels=np.arange(0.02,0.12,0.01)
    # color_limits = np.array([0.04,0.075,0.11])
    cmap_colors = ["blue","purple","orange"]
    
    cmap_norm=plt.Normalize(min(color_limits),max(color_limits))
    cmap_tuples = list(zip(map(cmap_norm,color_limits), cmap_colors))
    output_cmap = mcol.LinearSegmentedColormap.from_list("", cmap_tuples)
    cplot = ax.contour(X1,X2,Yp_frac,levels=contour_levels,cmap=output_cmap,norm=cmap_norm)
    ax.clabel(cplot, inline=1, fontsize=12)
    ax.set_xlabel('$\\phi$',size=12)
    ax.set_ylabel('$\\psi$',size=12)
    ax.grid(linestyle = '--', linewidth = 0.5)
    ax.set_title(f'$M_2={M_array[mi]:.2f}$',size=12)
    # ax.set_title(f'$C_0={Co_array[mi]:.2f}$',size=12)

fig.set_figwidth(8)
fig.set_figheight(3)
fig.tight_layout()
plt.show()

#     cplot = plt.contour(X1,X2,Yp_frac)
    
# plt.clabel(cplot, inline=1, fontsize=12)
# plt.xlabel('phi')
# plt.ylabel('psi')
# plt.title('Yp_frac')

# plt.show()

# Yp_stator_model.plot_accuracy(datate)
# Yp_rotor_model.plot_accuracy(datate)

# Yp_rotor_model.plot('phi','psi')



# fig,axes = plt.subplots(1,3,sharey=True,squeeze=True)

# n = 250
# PHI,PSI = np.meshgrid(np.linspace(0.5,1.1,n),np.linspace(1.2,2.2,n))
# # phi_vector = np.linspace(0.5,1.1,n)
# # psi_vector = np.linspace(1.2,2.2,n)
# phi_vector = PHI.ravel()
# psi_vector = PSI.ravel()

# # pitch_to_chord = turb.s_cx_stator.reshape(n,n)
# # eta_lost = turb.eta_lost.reshape(n,n)

# M_array = [0.55,0.7,0.85]

# for mi,ax in enumerate(axes):

#     opt_pitch_to_chord = np.zeros([n,n])

#     for i in range(n):
#         print(f'[{i}/{n}]')
#         for j in range(n):
#             phi=PHI[i][j]*np.ones(k)
#             psi=PSI[i][j]*np.ones(k)
#             M2=M_array[mi]*np.ones(k)
#             Co=np.linspace(0.4,0.8,k)

#             turb = TD.turbine(phi,
#                             psi,
#                             M2,
#                             Co)

#             pitch_to_chord = turb.s_cx_stator
#             eta_lost = turb.eta_lost
            
#             min_eta_lost = np.amin(eta_lost)
#             min_i = np.where(eta_lost == min_eta_lost)
#             opt_pitch_to_chord[i][j] = pitch_to_chord[min_i]


#             # plt.plot(pitch_to_chord,eta_lost,label=f'$\\psi$={psi_val}, $\\phi$={phi_val}')

#     # opt_pitch_to_chord=opt_pitch_to_chord.reshape(n,n)
#     contour_levels=[0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7]
#     color_limits = np.array([0.8,1.3,1.8])
#     cmap_colors = ["blue","purple","orange"]
#     cmap_norm=plt.Normalize(min(color_limits),max(color_limits))
#     cmap_tuples = list(zip(map(cmap_norm,color_limits), cmap_colors))
#     output_cmap = mcol.LinearSegmentedColormap.from_list("", cmap_tuples)
#     cplot = ax.contour(PHI,PSI,opt_pitch_to_chord,
#                         levels=contour_levels,cmap=output_cmap,norm=cmap_norm)
#     ax.clabel(cplot, inline=1, fontsize=12)
#     ax.set_xlabel('$\\phi$',size=12)
#     ax.set_ylabel('$\\psi$',size=12)
#     ax.grid(linestyle = '--', linewidth = 0.5)
#     ax.set_title(f'$M_2={M_array[mi]:.2f}$',size=12)
# # plt.title('s_cx')
# # plt.legend()
# plt.show()