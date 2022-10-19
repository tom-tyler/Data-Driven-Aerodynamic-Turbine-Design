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

matern_kernel = kernels.Matern(length_scale = (1,1,1,1,1),
                         length_scale_bounds=(1e-2,1e2),
                         nu=1.5
                         )

constant_kernel_1 = kernels.ConstantKernel(constant_value=1,
                                           constant_value_bounds=(1e-6,1e1)
                                           )

constant_kernel_2 = kernels.ConstantKernel(constant_value=1,
                                           constant_value_bounds=(1e-6,1e1)
                                           )

kernel = constant_kernel_1 * matern_kernel + constant_kernel_2

fit = fit_data(kernel_form=kernel,
            training_dataframe=training_data,
            CI_percent=10,
            number_of_restarts=20)

example_datapoint = pd.DataFrame({'phi':[0.8],
                                  'psi':[1.4],
                                  'Lambda':[0.5],
                                  'M':[0.6],
                                  'Co':[0.7]
                                  })

print(fit.predict(example_datapoint))

# limit_dict = {'phi':(0.5,0.7),
#               'psi':(1.3,1.5),
#               'Lambda':(0.5,0.58),
#               'M':(0.5,0.6),
#               'Co':(0.7,0.8)
#               }

# fit.find_global_max_min_values()
# print(fit.max_output_row)

# fit.plot_accuracy(testing_data)

# fit.plot_grid_vars(vary_var_1='phi',
#                    vary_or_constant_2='psi',
#                    column_var='Lambda',
#                    column_var_array=[0.45,0.5,0.55,0.6],
#                    row_var='Co',
#                    row_var_array=[0.6,0.7],
#                    constant_var='M',
#                    constant_var_value = 0.6,
#                    num_points=500
#                    )

# fit.plot_vars(phi=0.5,
#               psi=1.0,
#               Lambda='vary',
#               M='vary',
#               Co=0.6,
#               num_points=500,
#               efficiency_step=0.2
#               )