from dd_turb_design import fit_data, read_in_data, split_data, dim_2_nondim
import time
import numpy as np
import pandas as pd
# t1 = time.time()
# print("--- %s seconds ---" % (time.time() - t1))

data = read_in_data(state_retention_statistics=True,
                    ignore_incomplete=True)

traindf,testdf = split_data(data,
                            random_seed_state=0,
                            fraction_training=1.0)

fit = fit_data(training_dataframe=traindf,
               variables=['phi','psi','Po3_Po1','Co'],
               extra_variable_options=True)

print(fit.optimised_kernel)
# print(fit.scale)

# fit.plot_accuracy(testing_dataframe=testdf,
#                   line_error_percent=10,
#                   CI_percent=95,
#                   display_efficiency=False,
#                   identify_outliers=True,
#                   plot_errorbars=True)
fit.plot(x1='Po3_Po1',
         gridvars={},
         num_points=500,
         state_no_points=True)


# The culprit of bug is fitted_function
# robust scaling seemed to fix it

#need to fix limits of 1D plots with confidence intervals

#need to fix scaling
