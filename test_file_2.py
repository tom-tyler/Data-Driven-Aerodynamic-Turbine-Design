from dd_turb_design import fit_data, read_in_data, split_data

# data = read_in_data(dataset=['1D_lambda_data_1','1D_lambda_data_2']) #1 and 2 are very similar so only use 1
# data2d = read_in_data(dataset=['2D_phi_psi_data_1','2D_phi_psi_data_2','2D_phi_psi_data_3','2D_phi_psi_data_4'])
data = read_in_data(dataset='4D')
# data = read_in_data(dataset='5D')
# data = read_in_data()

traindf,testdf = split_data(data,
                            random_seed_state=0,
                            fraction_training=0.85)

fit = fit_data(training_dataframe=traindf,
               variables=['phi','psi','M','Co'])

# print(fit.nondim_to_dim(data))

# print(fit.optimised_kernel)

# fit.plot_vars(phi='vary',
#               psi='vary',
#               num_points=500,
#               efficiency_step=0.5,
#               plot_training_points=True,
#               CI_percent=95
#               )

# print(fit.find_global_max_min_values())

fit.plot_accuracy(testing_dataframe=testdf,
                  line_error_percent=10,
                  CI_percent=95,
                  display_efficiency=False,
                  identify_outliers=True,
                  plot_errorbars=True)

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