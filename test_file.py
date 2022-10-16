from signal import set_wakeup_fd
from dd_turb_design import fit_data
import pandas as pd
from sklearn.gaussian_process import kernels
import matplotlib.pyplot as plt
import numpy as np

training_file_path = 'Data-Driven-Aerodynamic-Turbine-Design\data-B for training.csv'
training_data = pd.read_csv(training_file_path, names=["phi", "psi", "Lambda", "M", "Co", "eta_lost"])

testing_file_path = 'Data-Driven-Aerodynamic-Turbine-Design\data-A for testing.csv'
testing_data = pd.read_csv(testing_file_path, names=["phi", "psi", "Lambda", "M", "Co", "eta_lost"])

min_rmse = 1


# make sure nu is one of [0.5, 1.5, 2.5, inf]
for exponent in [1.6,1.8,2.0,2.2,2.4]:
    kernel1 = kernels.Matern(length_scale = (1,1,1,1,1),
                                length_scale_bounds=(1e-4,1e4),
                                nu=1.5
                                )


    kernel = 0.05 * kernel1 ** exponent + 1.0
    
    fit = fit_data(kernel_form=kernel,
                training_dataframe=training_data,
                CI_percent=20,
                number_of_restarts=100)
    fit.predict(testing_data)
    
    rmse = fit.RMSE
    if rmse<min_rmse:
        min_rmse=rmse
        min_kernel = kernel
        
    print(exponent)

fit = fit_data(kernel_form=min_kernel,
            training_dataframe=training_data,
            CI_percent=20,
            number_of_restarts=100)
fit.predict(testing_data)

print(fit.RMSE)
print(fit.optimised_kernel)

fig,ax = plt.subplots(1,1,sharex=True,sharey=True)
plt.scatter(fit.output_array_test,fit.mean_prediction)
plt.plot(fit.output_array_test,fit.output_array_test)
plt.show()

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

# fig.suptitle("Data-driven turbine design")
# plt.show()