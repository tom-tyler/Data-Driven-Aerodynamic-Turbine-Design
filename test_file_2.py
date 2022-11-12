from dd_turb_design import fit_data, fix_vars, read_in_data
import pandas as pd
from sklearn.gaussian_process import kernels
import matplotlib.pyplot as plt
import numpy as np

# data_A = pd.read_csv('Data\data-A for testing.csv',
#                      names=["phi", "psi", "Lambda", "M", "Co", "eta_lost"]
#                      )
# data_B = pd.read_csv('Data\data-B for training.csv', 
#                      names=["phi", "psi", "Lambda", "M", "Co", "eta_lost"]
#                      )

# data_C = pd.read_csv('Data\data-C.csv', 
#                      names=["phi", "psi", "Lambda", "M", "Co", "eta_lost"]
#                      )
# data_D = pd.read_csv('Data\data-D.csv', 
#                      names=["phi", "psi", "Lambda", "M", "Co", "eta_lost"]
#                      )
# data_E = pd.read_csv('Data\data-E.csv', 
#                      names=["phi", "psi", "Lambda", "M", "Co", "eta_lost"]
#                      )
# data_F = pd.read_csv('Data\data-F.csv', 
#                      names=["phi", "psi", "Lambda", "M", "Co", "eta_lost"]
#                      )

# lambda_1d_data = pd.read_csv('Data\Lambda_1D.csv', 
#                      names=["phi", "psi", "Lambda", "M", "Co", "eta_lost",'runid']
#                      ).drop(columns=['runid'])

data_dict=read_in_data()
dataframe_list = data_dict.values()

data = pd.concat(dataframe_list,ignore_index=True)
# data = pd.concat([data_A],ignore_index=True)

# round_dict = {'phi': 2, 'psi': 2, 'Lambda': 2, 'M': 2, 'Co': 2, 'eta_lost':4}
# training_data = data.sample(frac=1.0,random_state=2)
# testing_data = data.loc[~data.index.isin(training_data.index)]

# fix_vars(lambda_1d_data,
#          vars_to_fix=['M','Co','phi','psi'],
#          values='mean')

fit = fit_data(training_dataframe=data,
               number_of_restarts=30)

print(fit.optimised_kernel)

fit.plot_vars(phi='mean',
              psi='mean',
              Lambda='vary',
              M='mean',
              Co='mean',
              num_points=500,
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

