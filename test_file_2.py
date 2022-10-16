from signal import set_wakeup_fd
from dd_turb_design import fit_data
import pandas as pd
from sklearn.gaussian_process import kernels
import matplotlib.pyplot as plt
import numpy as np

training_file_path = 'Data-Driven-Aerodynamic-Turbine-Design\data-B for training.csv'
training_data = pd.read_csv(training_file_path, 
                            names=["phi", "psi", "Lambda", "M", "Co", "eta_lost"]
                            )

testing_file_path = 'Data-Driven-Aerodynamic-Turbine-Design\data-A for testing.csv'
testing_data = pd.read_csv(testing_file_path, 
                           names=["phi", "psi", "Lambda", "M", "Co", "eta_lost"]
                           )

kernel1 = kernels.Matern(length_scale = (1,1,1,1,1),
                         length_scale_bounds=(1e-4,1e4),
                         nu=1.5
                         )

kernel = 0.025 * kernel1 + 0.1

fit = fit_data(kernel_form=kernel,
            training_dataframe=training_data,
            CI_percent=20,
            number_of_restarts=20)

fit.predict(testing_data)

fig,ax = plt.subplots(1,1,sharex=True,sharey=True)

fit.plot_accuracy(ax)

# fig, axes = plt.subplots(3,3,sharex=True,sharey=True)

# M_array = [0.6,0.7,0.8]
# Co_array = [0.6,0.7,0.8]

#values chosen to avoid extrapolation
# limit_dict = {'phi':(np.around(training_data['phi'].min(),decimals=1),np.around(training_data['phi'].max(),decimals=1)),
#             'psi':(np.around(training_data['psi'].min(),decimals=1),np.around(training_data['psi'].max(),decimals=1)),
#             'Lambda':(np.around(training_data['Lambda'].min(),decimals=1),np.around(training_data['Lambda'].max(),decimals=1)),
#             'M':(np.around(training_data['M'].min(),decimals=1),np.around(training_data['M'].max(),decimals=1)),
#             'Co':(np.around(training_data['Co'].min(),decimals=1),np.around(training_data['Co'].max(),decimals=1))}

# for (i,j), ax in np.ndenumerate(axes):

#     fit.plot_vars(limit_dict,
#                   ax,
#                   phi='vary',
#                   psi='vary',
#                   Lambda=0.5,
#                   M=M_array[i],
#                   Co=Co_array[j],
#                   num_points=1000,
#                   display_efficiency=True,
#                   efficiency_step=0.25,
#                   swap_axis=False
#                   )

fig.suptitle("Data-driven turbine design")
plt.show()