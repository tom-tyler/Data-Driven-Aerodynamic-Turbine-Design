from signal import set_wakeup_fd
from dd_turb_design import fit_data
import pandas as pd
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
import numpy as np

training_file_path = 'Data-Driven-Aerodynamic-Turbine-Design\data-B for training.csv'
training_data = pd.read_csv(training_file_path, names=["phi", "psi", "Lambda", "M", "Co", "eta_lost"])

testing_file_path = 'Data-Driven-Aerodynamic-Turbine-Design\data-A for testing.csv'
testing_data = pd.read_csv(testing_file_path, names=["phi", "psi", "Lambda", "M", "Co", "eta_lost"])

kernel = 0.05 * RBF(length_scale = (1.0,1.0,1.0,1.0,1.0), 
                    length_scale_bounds=(1e-4,1e2))

#values chosen to avoid extrapolation
limit_dict = {'phi':(np.around(training_data['phi'].min(),decimals=1),np.around(training_data['phi'].max(),decimals=1)),
              'psi':(np.around(training_data['psi'].min(),decimals=1),np.around(training_data['psi'].max(),decimals=1)),
              'Lambda':(np.around(training_data['Lambda'].min(),decimals=1),np.around(training_data['Lambda'].max(),decimals=1)),
              'M':(np.around(training_data['M'].min(),decimals=1),np.around(training_data['M'].max(),decimals=1)),
              'Co':(np.around(training_data['Co'].min(),decimals=1),np.around(training_data['Co'].max(),decimals=1))}

fit = fit_data(kernel_form=kernel,
               training_dataframe=training_data,
               CI_percent=20)

fig,(ax1) = plt.subplots(1,1,sharex=True,sharey=True)

fit.plot_vars(limit_dict,
              ax1,
              phi='vary',
              psi='vary',
              Lambda=0.4,
              M=0.84,
              Co=0.7,
              num_points=1000,
              display_efficiency=True,
              efficiency_step=0.5,
              swap_axis=True
              )
plt.show()