from dd_turb_design import fit_data, read_in_data, split_data, dim_2_nondim
import time
import numpy as np
import pandas as pd
# t1 = time.time()
# print("--- %s seconds ---" % (time.time() - t1))

data = read_in_data(state_retention_statistics=True)

traindf,testdf = split_data(data,
                            random_seed_state=0,
                            fraction_training=0.75)
#for final report, take 100 (for example) randome seeds, and
#and compute average R^2 and RMSE as these vary with random seed,
#(sensitive to what randome sample is taken)

fit = fit_data(training_dataframe=traindf,
               variables=['phi','psi','M2','Co'])

# print(fit.optimised_kernel)

fit.plot_accuracy(testing_dataframe=testdf,
                  line_error_percent=10)

# fit.plot(x1='phi',
#          x2='psi',
#          gridvars={},
#          num_points=500,
#          state_no_points=True)



#need to fix limits of 1D plots with confidence intervals

#need to fix scaling
