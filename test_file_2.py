from dd_turb_design import fit_data, read_in_data, split_data, dim_2_nondim
import time
import numpy as np
# t1 = time.time()
# print("--- %s seconds ---" % (time.time() - t1))

data = read_in_data('4D',state_retention_statistics=True)

traindf,testdf = split_data(data,
                            random_seed_state=0,
                            fraction_training=1.0)

fit = fit_data(training_dataframe=traindf,
               variables=['M2','psi','Co','phi'])

print(fit.optimised_kernel)

# print(nondim_stage_from_Lam(data))

# print(fit.find_global_max_min_values())

# fit.plot_accuracy(testing_dataframe=testdf,
#                   line_error_percent=10,
#                   CI_percent=95,
#                   display_efficiency=False,
#                   identify_outliers=True,
#                   plot_errorbars=True)

# fit.plot(x1='Co',
#          gridvars={'phi':[0.5,0.8,1.1]},
#          rotate_grid=True,
#          num_points=500,
#          constants={'M2':0.7,
#                     'psi':1.8},
#          plot_actual_data=True,
#          show_actual_with_model=False)

fit.plot(x1='Co',
         gridvars={'psi':[1.2,1.6,2.0]},
         rotate_grid=True,
         num_points=500,
         constants={'M2':0.7,
                    'phi':0.8},
         plot_actual_data=True,
         show_actual_with_model=False)

# print(dim_2_nondim(shaft_power=20e6,
#                     stagnation_pressure_ratio=2.0,
#                     blade_number=40))