from dd_turb_design import fit_data, read_in_data, split_data, dim_2_nondim
import time
import numpy as np
import pandas as pd
# t1 = time.time()
# print("--- %s seconds ---" % (time.time() - t1))

data = read_in_data(state_retention_statistics=True)

traindf,testdf = split_data(data,
                            random_seed_state=0,
                            fraction_training=0.6)
#for final report, take 100 (for example) randome seeds, and
#and compute average R^2 and RMSE as these vary with random seed,
#(sensitive to what randome sample is taken)

fit = fit_data.no_points
print(fit)

# fit = fit_data(training_dataframe=traindf,
#                variables=['phi','psi','M2','Co'],
#                limit_dict={'phi':(0.5,1.0),
#                            'psi':(1.0,2.0),
#                            'M2':(0.5,0.85),
#                            'Co':(0.55,0.7)})

# print(fit.optimised_kernel)
# print(fit.find_global_max_min_values(30))

# fit.plot_accuracy(testing_dataframe=testdf,
#                   line_error_percent=10)

# fit.plot(x1='Co',
#          constants={'psi':1.28,
#                    'M2':0.7,
#                    'phi':0.8},
#          rotate_grid=True,
#          num_points=500)

# generation
# Ma2_lim = (0.5, 0.95)
# phi_lim = (0.5, 1.2)
# psi_lim = (1.0, 2.4)
# Lam_ref = 0.5
# Co_lim = (0.4, 0.8)

# gridvars
# {'M2':[0.55,0.6,0.7,0.8,0.9],'psi':[1.1,1.4,1.8,2.0,2.3]}

# fit.plot(optimum_plot=True,
#          x1='phi',
#          x2='psi',
#          gridvars={'M2':[0.55,0.6,0.7,0.8,0.9],'Co':[0.55,0.6,0.65,0.7]},
#          num_points=500,
#          plot_actual_data_filter_factor=15)

# fit.plot(optimum_plot=True,
#          x1='psi',
#          x2='phi',
#          gridvars={'Co':[0.55,0.6,0.65,0.7],'M2':[0.55,0.6,0.7,0.8,0.9]},
#          num_points=500,
#          plot_actual_data_filter_factor=15)

# fit.plot(optimum_plot=True,
#          x1='M2',
#          x2='phi',
#          gridvars={'Co':[0.55,0.6,0.65,0.7],'psi':[1.1,1.3,1.5,1.7,1.9]},
#          num_points=500,
#          plot_actual_data_filter_factor=15)

# fit.plot(x1='phi',
#          x2='psi',
#          constants={'Co':0.66,
#                     'M2':0.7},
#          num_points=400,
#          efficiency_step=0.25)

#need to fix limits of 1D plots with confidence intervals

#need to fix scaling
