from turbine_design import turbine_design as TD
from turbine_design import data_tools as tools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# data = tools.read_in_data()

# data = tools.read_in_large_dataset('bonus',state_retention_statistics=True,factor=3)

# print(data[['phi','psi','Co','Lambda','M2','eta_lost']])
turb_1 = TD.turbine(phi = 0.6,
                 psi = 1.4,
                 M2  = 0.5,
                 Co  = 0.65)

turb_1.get_dimensional(Omega = 314,
                       Po1   = 160000,
                       To1   = 1800)

turb_1.get_blade("2D")

# turb_1.get_blade("3D")

# turb = TD.turbine(phi=0.6,
#                   psi=1.4,
#                   M2=0.5,
#                   Co=0.65)
# turb = TD.turbine(phi=0.6,
#                   psi=2.0,
#                   M2=0.9,
#                   Co=0.65)

# turb.get_blade(2,stack_ratios=[5,4])

# turb.get_blade(3)


# import numpy as np
# from sklearn.metrics import r2_score 
# n = 100
# X1,X2 = np.meshgrid(np.linspace(0.5,1.1,n),np.linspace(1.2,2.2,n))
# X1_vector = X1.ravel() #vector of "all" x coordinates from meshgrid
# X2_vector = X2.ravel() #vector of "all" y coordinates from meshgrid
         

# turb = TD.turbine(X1_vector,
#                   X2_vector,
#                   0.7*np.ones(len(X1_vector)),
#                   0.65*np.ones(len(X1_vector)))

# turb.get_nondim()

# output = turb.Yp_rotor.reshape(n,n)


# cplot = plt.contour(X1,X2,output)
# plt.clabel(cplot, inline=1, fontsize=14)
# plt.xlabel('phi')
# plt.ylabel('psi')
# plt.title('Yp_rotor')

# plt.show()

# turb.get_non_dim_geometry()
# n = 100
# co_vector = np.linspace(0.45,0.75,n)

         
# turb = TD.turbine([0.81,0.7],[1.65,1.5],[0.72,0.71],[0.67,0.65])
# turb = TD.turbine(0.8*np.ones(len(co_vector)),
#                   1.8*np.ones(len(co_vector)),
#                   0.7*np.ones(len(co_vector)),
#                   co_vector)

# turb.dim_from_omega(314,1800,160000)

# no_blades = turb.num_blades_stator
# eta = turb.eta

# # create figure and axis objects with subplots()
# fig,ax = plt.subplots()
# # make a plot
# ax.plot(co_vector,
#         no_blades,
#         color="darkorange")
# # set x-axis label
# ax.set_xlabel("$C_0$", fontsize = 12)
# # set y-axis label
# ax.set_ylabel("No. blades",
#               color="darkorange",
#               fontsize=12)
# ax.set_xlim(0.45,0.75)
# ax.set_title('$\\phi=0.8$, $\\psi=1.8$, $\\Lambda=0.5$, $M_2=0.7$')

# # twin object for two different y-axis on the sample plot
# ax2=ax.twinx()
# # make a plot with different y-axis using second axis object
# ax2.plot(co_vector, eta,color="cornflowerblue")
# ax2.set_ylabel("$\\eta$ (%)",color="cornflowerblue",fontsize=12)

# max_eta = np.amax(eta)

# i = np.where(eta==max_eta)

# no_opt = no_blades[i]

# ax.axvline(co_vector[i],0.314,ymax=0.958,linewidth=1, color='r',dashes=(5, 2, 1, 2))
# ax.axhline(no_opt,xmin=0,xmax=co_vector[i],linewidth=1, color='r',dashes=(5, 2, 1, 2))
# ax.text(0.52,no_opt,f'{int(no_opt)} blades',size=12, color='r',
#         horizontalalignment='left')

# plt.show()

# data = tools.read_in_data('5D',state_retention_statistics=True)

# # data = tools.read_in_large_dataset('5D',state_retention_statistics=True)

# datatr,datate = tools.split_data(data)

# model = TD.turbine_GPR('eta_lost_5D')

# model.fit(datatr,
#           variables=['phi','psi','Co','M2','Lambda'],
#           output_key='eta_lost',
#           model_name='eta_lost_5D')

# print(model.optimised_kernel)

# model.plot_accuracy(datate,
#                     line_error_percent=10,
#                     legend_outside=True,
#                     identify_outliers=True)

# fig,axes = model.plot(x1='phi',x2='psi',
#            CI_percent=0,
#            constants={'M2':0.67,
#                       'Co':0.65,
#                       'Lambda':0.50},
#            num_points=500,
#            contour_step=0.0025,
#            show=False)

# axes.scatter([0.65,0.95,0.81,0.81,0.81],[1.78,1.78,1.78,1.20,1.50],marker='x',color='magenta')
# axes.set_xlim(0.6,1.0)
# axes.set_ylim(1.0,2.0)
# plt.show()

# eta_model = model

# print(eta_model.predict(pd.DataFrame({'phi':[8.169344704738479290e-01],
#                                      'psi':[2.210212160066015397e+00],
#                                      'Co':[6.500155329704284668e-01],
#                                      'M2':[6.132978689012978935e-01],
#                                      'eta':[(1-6.046904698383070986e-02)]}),
#                         True,
#                         True))
# print(1-eta_model.upper,1-eta_model.lower,eta_model.std_prediction,1-eta_model.mean_prediction)

# eta_model = TD.turbine_GPR('eta_lost')

# print(eta_model.predict(pd.DataFrame({'phi':[8.169344704738479290e-01],
#                                      'psi':[2.210212160066015397e+00],
#                                      'Co':[6.500155329704284668e-01],
#                                      'M2':[6.132978689012978935e-01],
#                                      'eta_lost':[6.046904698383070986e-02]}),
#                         True,
#                         True))
# print(eta_model.lower,eta_model.upper,eta_model.std_prediction,eta_model.mean_prediction)


# print(eta_model.find_max_min())

# eta_model.plot('phi',
#                'psi',
#                gridvars={'Co':(0.6,0.65,0.7),
#                          'M2':(0.6,0.7,0.8)},
#                num_points=200)







# eta_lost = turbine_GPR('eta_lost')

# eta_lost.plot(x1='phi',
#               x2='psi',
#               contour_step=0.002,
#               num_points=250,
#               CI_percent=0,
#               show_min=True)

# data = read_in_large_dataset(state_retention_statistics=True)
# # eta_lost.plot_accuracy(data)


# rand_data,other=split_data(data,1.0,random_seed_state=9)

# turb = turbines(rand_data['phi'],rand_data['psi'],rand_data['M2'],rand_data['Co'])

# turb.get_eta_lost()
# x = turb.eta_lost
# y = rand_data['eta_lost']
# plt.scatter(x,y,marker='x')
# plt.plot(x,x,color='r')
# plt.show()



# R_square = r2_score(x, y) 
# print('Coefficient of Determination', R_square) 