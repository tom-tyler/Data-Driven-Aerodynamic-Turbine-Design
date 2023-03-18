# from turbine_design.turbine_design import turbine
# import matplotlib.pyplot as plt
# from turbine_design.data_tools import *
from turbine_design import turbine_design as TD
from turbine_design import data_tools as tools
import matplotlib.pyplot as plt
import numpy as np

turb = TD.turbine(phi=0.8,
                  psi=1.6,
                  M2=0.65,
                  Co=0.65)

turb.get_blade_3D()

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