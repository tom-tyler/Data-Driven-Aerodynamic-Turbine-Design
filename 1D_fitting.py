"""goal is to create a ML model for a single input function"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats.qmc import LatinHypercube

def f(x):
    return 0.5*X**3 - X**2 - 20*X

fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,sharex=True,sharey=True)

scope = 1
training_data_size = 2
grid_range = 10
grid_size = 1000
kernel = 1.0 * RBF(length_scale=1.0,length_scale_bounds=(1e-5,400))

for ax in [ax1,ax2,ax3,ax4]:

    training_data_size+=2

    x = np.linspace(start=-1*grid_range, stop=grid_range, num=grid_size)
    X = x.reshape(-1, 1)
    y = np.squeeze(f(X))

    sampler = LatinHypercube(d=1)
    training_indices = (len(y)*scope*sampler.random(n=training_data_size)).astype(int).reshape(training_data_size)
    X_train, y_train = X[training_indices], y[training_indices]

    #max and min length scale bounds should be considered very appropriately based on grid size
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=30,alpha=0.1)
    gaussian_process.fit(X_train, y_train)
    
    mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)
    upper_confidence_interval = mean_prediction - 1.96 * std_prediction
    lower_confidence_interval = mean_prediction + 1.96 * std_prediction



    ax.plot(X, y, label=r"$f(x)$", linestyle="dotted")
    ax.scatter(X_train,y_train, label=r"$Samples$", marker="x")
    ax.plot(X, mean_prediction, label="Mean prediction")
    ax.fill_between(
        x=X.ravel(),
        y1=upper_confidence_interval,   # 95% of area is within 1.96x standard deviation of the mean
        y2=lower_confidence_interval,
        alpha=0.5,                                       #transparency
        label=r"95% confidence interval"
    )
    ax.legend()
    #ax.set_xlabel("$x$")
    #ax.set_ylabel("$f(x)$")
    ax.set_title(f'n={training_data_size}')
    
fig.suptitle("Gaussian process regression on 1D function")
plt.show()