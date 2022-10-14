from dd_turb_design import fit_data
import pandas as pd
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt


training_file_path = 'Data-Driven-Aerodynamic-Turbine-Design\data-B for training.csv'
training_data = pd.read_csv(training_file_path, names=["phi", "psi", "Lambda", "M", "Co", "eta_lost"])

testing_file_path = 'Data-Driven-Aerodynamic-Turbine-Design\data-A for testing.csv'
testing_data = pd.read_csv(testing_file_path, names=["phi", "psi", "Lambda", "M", "Co", "eta_lost"])

kernel = 0.05 * RBF(length_scale = (1.0,1.0,1.0,1.0,1.0), length_scale_bounds=(1e-4,1e2))

fit = fit_data(kernel_form=kernel,training_dataframe=training_data)

mean_prediction = fit.predict(testing_dataframe=testing_data)
