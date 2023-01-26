from dd_turb_design import fit_data, read_in_data, split_data, fix_vars
import numpy as np

# data_1D = read_in_data(dataset=['1D_lambda_data_2']) #1 and 2 are very similar so only use 1

# data_2D = read_in_data(dataset=['2D_phi_psi_data_1',
#                                 '2D_phi_psi_data_2',
#                                 '2D_phi_psi_data_3',
#                                 '2D_phi_psi_data_4'])

data_4D = read_in_data(dataset='4D')

# data_5D = read_in_data(dataset='5D')

# data = read_in_data()

# data_1D = fix_vars(data_1D,
#                    ['phi','psi','M','Co'],
#                    [0.81,1.78,0.7,0.65])

# data_2D = fix_vars(data_2D,
#                    ['Lambda','M','Co'],
#                    [0.51,0.66,0.66])

data_4D = fix_vars(data_4D,
                   ['Lambda'],
                   [0.5])

traindf,testdf = split_data(data_4D,
                            random_seed_state=0,
                            fraction_training=1.0) #7 good results

fit = fit_data(training_dataframe=traindf,
               nu=np.inf,
               scale_name='minmax',
               number_of_restarts=30)

#need noise if rbf kernel. trade off between noise and nu

print(fit.optimised_kernel)

# fit.plot_vars(Co='vary',
#               Lambda=0.5,
#               psi=1.78,
#               M=0.7,
#               phi=0.81,
#               num_points=300,
#               efficiency_step=0.3,
#               plot_training_points=False,
#               CI_percent=95
#               )

fit.plot_vars(phi='vary',
              psi='vary',
              num_points=500,
              efficiency_step=0.5,
              plot_training_points=True,
              CI_percent=95
              )


# fit.plot_accuracy(testing_dataframe=testdf,
#                   line_error_percent=10,
#                   CI_percent=95,
#                   display_efficiency=False,
#                   identify_outliers=True,
#                   plot_errorbars=True)

# fit.plot_grid_vars('M',
#                    'phi',
#                    'Lambda',
#                    [0.45,0.5,0.55],
#                    'psi',
#                    [1.5,2.0],
#                    'Co',
#                    0.66,
#                    num_points=400,
#                    CI_percent=70)

# fit.plot_grid_vars('phi',
#                    'psi',
#                    'Lambda',
#                    [0.4,0.5,0.6],
#                    'M',
#                    [0.5,0.7,0.9],
#                    'Co',
#                    0.66,
#                    num_points=400,
#                    CI_percent=70)