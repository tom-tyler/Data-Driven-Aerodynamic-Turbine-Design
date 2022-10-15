from dd_turb_design import fit_data
import pandas as pd
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt


training_file_path = 'Data-Driven-Aerodynamic-Turbine-Design\data-B for training.csv'
training_data = pd.read_csv(training_file_path, names=["phi", "psi", "Lambda", "M", "Co", "eta_lost"])

testing_file_path = 'Data-Driven-Aerodynamic-Turbine-Design\data-A for testing.csv'
testing_data = pd.read_csv(testing_file_path, names=["phi", "psi", "Lambda", "M", "Co", "eta_lost"])

kernel = 0.05 * RBF(length_scale = (1.0,1.0,1.0,1.0,1.0), 
                    length_scale_bounds=(1e-4,1e2))

#values chosen to avoid exprapolation
limit_dict = {'phi':(0.4,1.2),'psi':(1,2.6),'Lambda':(0.4,0.6),'M':(0.54,0.93),'Co':(0.53,0.8)}

fit = fit_data(kernel_form=kernel,
               training_dataframe=training_data,
               confidence_scalar=1.96)

fig,(ax1) = plt.subplots(1,1,sharex=True,sharey=True)
fit.plot_vars(limit_dict,
              ax1,
              phi=0.63,
              psi='vary',
              Lambda=0.4,
              M=0.84,
              Co=0.55,
              num_points=200,
              display_efficiency=True,
              efficiency_step=0.5
              )
plt.show()