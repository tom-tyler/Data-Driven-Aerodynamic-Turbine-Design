# from turbine_design.turbine_design import turbine
# import matplotlib.pyplot as plt
# from turbine_design.data_tools import *
from turbine_design import turbine_design as TD

# import numpy as np
# from sklearn.metrics import r2_score 

turb = TD.turbine(0.81,1.65,0.72,0.67)

turb.get_non_dim_geometry()

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