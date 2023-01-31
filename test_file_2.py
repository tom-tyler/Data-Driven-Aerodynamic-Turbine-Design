from dd_turb_design import fit_data, read_in_data, split_data, nondim_stage_from_Lam

data = read_in_data('5D only')

traindf,testdf = split_data(data,
                            random_seed_state=0,
                            fraction_training=1.0)

fit = fit_data(training_dataframe=traindf,
               variables=['phi','psi','Lambda','M','Co'])

print(fit.optimised_kernel)

# print(nondim_stage_from_Lam(data))

# fit.plot_vars(phi='vary',
#               num_points=500,
#               efficiency_step=0.1,
#               plot_training_points=True,
#               CI_percent=95,
#               superpose_variable='T1_To1'
#               )

fit.plot_vars(x='phi',
              z2='M1',
              constants={'psi':'mean',
                         'Lambda':'mean',
                         'M':'mean',
                         'Co':'mean'},
              num_points=400)

# print(fit.find_global_max_min_values())

# fit.plot_accuracy(testing_dataframe=testdf,
#                   line_error_percent=10,
#                   CI_percent=95,
#                   display_efficiency=False,
#                   identify_outliers=True,
#                   plot_errorbars=True)

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