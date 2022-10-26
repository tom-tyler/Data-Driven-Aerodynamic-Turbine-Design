from signal import set_wakeup_fd
from dd_turb_design import fit_data
import pandas as pd
from sklearn.gaussian_process import kernels
import matplotlib.pyplot as plt
import numpy as np

data_A = pd.read_csv('Data-Driven-Aerodynamic-Turbine-Design\data-A for testing.csv',
                     names=["phi", "psi", "Lambda", "M", "Co", "eta_lost"]
                     )
data_B = pd.read_csv('Data-Driven-Aerodynamic-Turbine-Design\data-B for training.csv', 
                     names=["phi", "psi", "Lambda", "M", "Co", "eta_lost"]
                     )
data_C = pd.read_csv('Data-Driven-Aerodynamic-Turbine-Design\data-C.csv', 
                     names=["phi", "psi", "Lambda", "M", "Co", "eta_lost"]
                     )

data = pd.concat([data_A,data_B,data_C],ignore_index=True)
data=data_C
round_dict = {'phi': 2, 'psi': 2, 'Lambda': 2, 'M': 2, 'Co': 2, 'eta_lost':4}
training_data = data.sample(frac=1.0,random_state=2)
testing_data = data.loc[~data.index.isin(training_data.index)]

matern_kernel = kernels.Matern(length_scale = (1,1,1,1,1),
                         length_scale_bounds=(1e-2,1e2),
                         nu=1.5
                         )

rbf_kernel_1d = kernels.RBF(length_scale=(1),
                         length_scale_bounds=(1e-3,1e4)
                         )
rbf_kernel_5d = kernels.RBF(length_scale=(1,1,1,1,1),
                         length_scale_bounds=(1e-1,1e4)
                         )

constant_kernel_1 = kernels.ConstantKernel(constant_value=1,
                                           constant_value_bounds=(1e-8,1e1)
                                           )

constant_kernel_2 = kernels.ConstantKernel(constant_value=1,
                                           constant_value_bounds=(1e-6,1e1)
                                           )

kernel = constant_kernel_1 * matern_kernel + constant_kernel_2

fit = fit_data(kernel_form=kernel,
            training_dataframe=training_data,
            number_of_restarts=20)


# example_datapoint = pd.DataFrame({'phi':[0.8],
#                                   'psi':[1.4],
#                                   'Lambda':[0.5],
#                                   'M':[0.6],
#                                   'Co':[0.7]
#                                   })
# print(fit.predict(example_datapoint))

# limit_dict = {'phi':(0.5,0.7),
#               'psi':(1.3,1.5),
#               'Lambda':(0.5,0.58),
#               'M':(0.5,0.6),
#               'Co':(0.7,0.8)
#               }

# fit.find_global_max_min_values()
# print(fit.max_output_row)

# fit.plot_accuracy(testing_data,
#                   line_error_percent=1,
#                   CI_percent=95,
#                   display_efficiency=True)

# fit.plot_grid_vars(vary_var_1='phi',
#                    vary_or_constant_2='Lambda',
#                    column_var='psi',
#                    column_var_array=[1.0,1.5,2.0,2.5],
#                    row_var='M',
#                    row_var_array=[0.65,0.7,0.75],
#                    constant_var='Co',
#                    constant_var_value = 0.7,
#                    num_points=500,
#                    efficiency_step=0.5
#                    )


# fit.plot_vars(phi='vary',
#               psi='mean',
#               Lambda='mean',
#               M='mean',
#               Co='mean',
#               num_points=500,
#               efficiency_step=0.2,
#               plot_training_points=True,
#               CI_percent=95,
#               legend_outside=True,
#               display_efficiency=False
#               )

fit.plot_vars(phi='vary',
              psi='vary',
              Lambda=5.073657893138810993e-01,
              M=6.471594517666803270e-01,
              Co=6.840099096298217773e-01,
              num_points=500,
              efficiency_step=0.5,
              plot_training_points=True,
              CI_percent=95,
              legend_outside=True,
              display_efficiency=True
              )
