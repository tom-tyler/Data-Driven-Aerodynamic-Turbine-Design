from turbine_design import turbine_design as TD
from turbine_design import data_tools as tools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = tools.read_in_large_dataset()

datatr,datate = tools.split_data(data)

model = TD.turbine_GPR()
model.fit(datatr,
          variables=['phi','psi','M2','Co'],
          output_key='eta_lost',
          length_bounds=[1e-2,1e3],
          noise_magnitude=1e-6,
          noise_bounds=[1e-20,1e-1],
          number_of_restarts=3)

model.plot('phi',
           'psi',
           gridvars={'Co':(0.6,0.7),
                     'M2':(0.7,0.85)},
           num_points=100)

print(model.optimised_kernel)

# model.plot_accuracy(datate)

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

# turb = TD.turbine(phi=0.6,
#                   psi=2.0,
#                   M2=0.9,
#                   Co=0.65)

# turb.get_blade_3D()

# turb.get_blade_2D()


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
# co_vector = np.linspace(0.55,0.7,n)

         
# # turb = TD.turbine([0.81,0.7],[1.65,1.5],[0.72,0.71],[0.67,0.65])
# turb = TD.turbine(0.8*np.ones(len(co_vector)),
#                   0.7*np.ones(len(co_vector)),
#                   0.7*np.ones(len(co_vector)),
#                   co_vector)

# turb.dim_from_omega(314,1800,160000)

# output = turb.num_blades_stator
# plt.plot(co_vector,output)
# plt.xlabel('Co')
# plt.ylabel('No. blades')
# plt.grid()

# plt.show()


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