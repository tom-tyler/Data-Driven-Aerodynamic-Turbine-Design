from dd_turb_design import fit_data, read_in_data, split_data
import numpy as np
import sklearn.preprocessing as pre


data_2D = read_in_data(dataset=['2D_phi_psi_data_1',
                                '2D_phi_psi_data_2',
                                '2D_phi_psi_data_3',
                                '2D_phi_psi_data_4'])

data_1D = read_in_data(dataset=['1D_lambda_data_1',
                                '1D_lambda_data_2'])

data_5D = read_in_data(dataset='5D')
# print(data_5D)

# 5D lims:
# Ma2_lim = (0.5, 0.95)
# phi_lim = (0.5, 1.2)
# psi_lim = (1.0, 2.4)
# Lam_lim = (0.4, 0.6)
# Co_lim = (0.4, 0.8)

data = read_in_data()

traindf,testdf = split_data(data,
                            random_seed_state=11,
                            fraction_training=0.75) #7 good results

fit = fit_data(training_dataframe=traindf,
               nu=1.5,
               scale_name='robust')

#need noise if rbf kernel. trade off between noise and nu

print(fit.optimised_kernel)

fit.plot_vars(phi='vary',
              psi='vary',
              num_points=200,
              efficiency_step=0.5,
              plot_training_points=False,
              CI_percent=95
              )


fit.plot_accuracy(testing_dataframe=testdf,
                  line_error_percent=2,
                  CI_percent=95,
                  display_efficiency=True)

# fit.plot_grid_vars('phi',
#                    'psi',
#                    'Co',
#                    [0.5,0.6,0.7],
#                    'M',
#                    [0.6,0.7,0.8],
#                    'Lambda',
#                    0.5,
#                    num_points=400,
#                    CI_percent=70)