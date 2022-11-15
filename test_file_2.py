from dd_turb_design import fit_data, fix_vars, read_in_data
import pandas as pd
from sklearn.gaussian_process import kernels
import matplotlib.pyplot as plt
import numpy as np

data = read_in_data(dataset=['5D_turbine_data_1',
                             '5D_turbine_data_2',
                             '5D_turbine_data_3',
                             '5D_turbine_data_4',
                             '5D_turbine_data_5'])

data = read_in_data()


fit = fit_data(training_dataframe=data,
               number_of_restarts=20,
               noise_magnitude=1e-8)

fit.plot_vars(phi='vary',
              M=0.9,
              psi='vary',
              num_points=250,
              efficiency_step=0.5,
              plot_training_points=True,
              CI_percent=95,
              legend_outside=True,
              display_efficiency=True
              )

# fit.plot_accuracy(testing_data,
#                 line_error_percent=5,
#                 CI_percent=95,
#                 display_efficiency=False)

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


# fit.plot_grid_vars(vary_var_1='phi',
#                    vary_or_constant_2='Co',
#                    column_var='M',
#                    column_var_array=[0.5,0.6,0.7,0.8,0.9],
#                    row_var='psi',
#                    row_var_array=[2.0],
#                    constant_var='Lambda',
#                    constant_var_value = 0.5,
#                    num_points=500,
#                    efficiency_step=0.25,
#                    CI_percent=10
#                    )

# fit.plot_vars(phi='mean',
#               psi='mean',
#               Lambda='mean',
#               M='mean',
#               Co='vary',
#               num_points=500,
#               efficiency_step=0.2,
#               plot_training_points=True,
#               CI_percent=95,
#               legend_outside=True,
#               display_efficiency=False
#               )

