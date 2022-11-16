from dd_turb_design import fit_data, read_in_data, split_data
import pandas as pd
from sklearn.gaussian_process import kernels
import matplotlib.pyplot as plt
import numpy as np

data = read_in_data(dataset='all')
traindf,testdf = split_data(data)

fit = fit_data(training_dataframe=traindf,
               number_of_restarts=10)

#need noise if rbf kernel. trade off between noise and nu

# print(fit.optimised_kernel)

# fit.plot_vars(phi='vary',
#               psi='vary',
#               num_points=500,
#               efficiency_step=0.2,
#               plot_training_points=False,
#               CI_percent=80
#               )


fit.plot_accuracy(testing_dataframe=testdf,
                  line_error_percent=2,
                  CI_percent=95,
                  display_efficiency=True)
