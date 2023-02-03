from dd_turb_design import fit_data, read_in_data, split_data, dim_2_non_dim
import time
import numpy as np
# t1 = time.time()
# print("--- %s seconds ---" % (time.time() - t1))

data = read_in_data('4D',state_retention_statistics=True)

traindf,testdf = split_data(data,
                            random_seed_state=0,
                            fraction_training=1.0)

fit = fit_data(training_dataframe=traindf,
               variables=['phi','psi','M2','Co'])

print(fit.optimised_kernel)

# print(nondim_stage_from_Lam(data))

# print(fit.find_global_max_min_values())

# fit.plot_accuracy(testing_dataframe=testdf,
#                   line_error_percent=10,
#                   CI_percent=95,
#                   display_efficiency=False,
#                   identify_outliers=True,
#                   plot_errorbars=True)

fit.plot(x1='Co',
         x2='psi',
         num_points=400)

# print(dim_2_non_dim(shaft_power=20e6,
#                     stagnation_pressure_ratio=2.0,
#                     blade_number=40))