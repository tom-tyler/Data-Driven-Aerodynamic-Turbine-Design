from dd_turb_design import fit_data
import pandas as pd
from sklearn.gaussian_process import kernels
import matplotlib.pyplot as plt
import numpy as np

data_C = pd.read_csv('data-C.csv', 
                     names=["phi", "psi", "Lambda", "M", "Co", "eta_lost"]
                     )

data_F = pd.read_csv('data-F.csv', 
                     names=["phi", "psi", "Lambda", "M", "Co", "eta_lost"]
                     )

training_data = pd.concat([data_C,data_F],ignore_index=True)

fit = fit_data(training_dataframe=training_data,
            number_of_restarts=20,
            noise_magnitude=1e-7,
            force_dimensions=['phi','psi'])

fit.plot_vars(phi='vary',
              psi='vary',
              Lambda=0.51,
              M=0.645,
              Co=0.67,
              num_points=500,
              efficiency_step=0.25,
              plot_training_points=True,
              CI_percent=95,
              legend_outside=True,
              display_efficiency=True
              )
