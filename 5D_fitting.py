import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# Data in csv files is in form:
# phi
# psi
# Lambda
# Mach number at stator exit
# circulation coefficient
# lost efficiency

training_file_path = 'Data-Driven-Aerodynamic-Turbine-Design\data-B for training.csv'
training_data = pd.read_csv(training_file_path, names=["phi", "psi", "Lambda", "M", "Co", "eta_lost"])

# Uppercase represents matrix, lowercase represents vector
X_train = training_data[['phi','psi','Lambda','M','Co']] #two brackets are needed so that the array is 2D
y_train = training_data['eta_lost']

smoothness_kernel = 1*RBF(length_scale = (1,1,1,1,1), length_scale_bounds=(1e-6,1e6))

noise_kernel = 1*WhiteKernel(noise_level=0.1**2, noise_level_bounds=(1e-5, 1e5))
                
kernel = noise_kernel + smoothness_kernel
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gaussian_process.fit(X_train, y_train)

print(gaussian_process.kernel_)


testing_file_path = 'Data-Driven-Aerodynamic-Turbine-Design\data-A for testing.csv'
testing_data = pd.read_csv(testing_file_path, names=["phi", "psi", "Lambda", "M", "Co", "eta_lost"])

X = testing_data[['phi','psi','Lambda','M','Co']] #two brackets are needed so that the array is 2D
y = testing_data['eta_lost']

mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

x_axis = np.arange(28)
plt.plot(x_axis,mean_prediction)
plt.plot(x_axis,y)
# # plt.scatter(X, y, label='testing')
# # plt.scatter(X_train, y_train, label="training")
# #plt.plot(X, mean_prediction, label="Mean prediction")
# # plt.fill_between(
# #     X.to_numpy().ravel(),
# #     mean_prediction - 1.96 * std_prediction,
# #     mean_prediction + 1.96 * std_prediction,
# #     alpha=0.5,
# #     label=r"95% confidence interval",
# # )
# # plt.legend()
# # plt.xlabel("$x$")
# # plt.ylabel("$f(x)$")
# # _ = plt.title("Gaussian process regression on noise-free dataset")
plt.show()
