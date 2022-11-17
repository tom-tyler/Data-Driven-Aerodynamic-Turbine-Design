from dd_turb_design import fit_data, read_in_data, split_data

data_2D = read_in_data(dataset=['2D_phi_psi_data_1',
                             '2D_phi_psi_data_2',
                             '2D_phi_psi_data_3',
                             '2D_phi_psi_data_4'])
data_5D = read_in_data(dataset='5D')
data = read_in_data()
traindf,testdf = split_data(data,
                            random_seed_state=11,
                            fraction_training=0.75) #7 good results

fit = fit_data(training_dataframe=traindf,
               number_of_restarts=20,
               nu=1.5)

#need noise if rbf kernel. trade off between noise and nu

print(fit.optimised_kernel)

# fit.plot_vars(phi='vary',
#               psi='vary',
#               num_points=800,
#               efficiency_step=0.5,
#               plot_training_points=False,
#               CI_percent=95
#               )


fit.plot_accuracy(testing_dataframe=testdf,
                  line_error_percent=1,
                  CI_percent=95,
                  display_efficiency=True)

# fit.plot_grid_vars('phi',
#                    'psi',
#                    'Co',
#                    [0.7,0.75,0.8],
#                    'Lambda',
#                    [0.5],
#                    'M',
#                    0.7,
#                    num_points=400,
#                    CI_percent=0)