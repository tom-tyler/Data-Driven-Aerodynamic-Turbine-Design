"""goal is to create a ML model for a 2 input function (hyperbolic paraboloid)"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

x = np.linspace(start=-10, stop=10, num=100)
y = np.linspace(start=-10, stop=10, num=100)
print(np.dstack((x,y)))
X,Y = np.meshgrid(x,y)
#X = x.reshape(-1, 1)
z = X**2 - Y**2
plt.contourf(X,Y,z)
plt.show()

# training_indices = np.random.randint(low=len(z),size=6)
# X_train, Y_train, z_train = X[training_indices], Y[training_indices], z[training_indices]

# kernel = 1.0 * RBF(length_scale=(1.0,1.0)
# gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
# gaussian_process.fit(X_train, y_train)
# print(gaussian_process.kernel_)
# mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)
# upper_confidence_interval = mean_prediction - 1.96 * std_prediction
# lower_confidence_interval = mean_prediction + 1.96 * std_prediction


# plt.plot(X, y, label=r"$f(x) = x^{2}$", linestyle="dotted")
# plt.scatter(X_train,y_train, label=r"$Samples$", marker="x")
# plt.plot(X, mean_prediction, label="Mean prediction")
# plt.fill_between(
#     x=X.ravel(),
#     y1=upper_confidence_interval,   # 95% of area is within 1.96x standard deviation of the mean
#     y2=lower_confidence_interval,
#     alpha=0.5,                                       #transparency
#     label=r"95% confidence interval"
# )
# plt.legend()
# plt.xlabel("$x$")
# plt.ylabel("$f(x)$")
# plt.title("Gaussian process regression on quadratic function (zero noise)")
# plt.show()