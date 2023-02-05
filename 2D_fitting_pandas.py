"""goal is to create a ML model for a 2 input function"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from matplotlib.colors import ListedColormap
from scipy.stats.qmc import LatinHypercube
from sklearn.metrics import mean_squared_error

def f(x,y):
    # return x**2 - y**2
    return (x-3)**2 + x*y + (y+4)**2

grid_size=1000
training_data_size=2
num_contours = 7
grid_range = 10
scope = 1
kernel = 1.0 * RBF(length_scale=(1.0,1.0), length_scale_bounds=(1e-5,100))
alpha_value = 0.1
hyperparameter_iterations = 30

fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,sharex=True,sharey=True)

sampler = LatinHypercube(d=1)
x = np.linspace(start=-1*grid_range, stop=grid_range, num=grid_size)
y = np.linspace(start=-1*grid_range, stop=grid_range, num=grid_size)
X,Y = np.meshgrid(x,y) # creates two matrices which vary across in x and y
X_vector = X.ravel() #vector of "all" x coordinates from meshgrid
Y_vector = Y.ravel() #vector of "all" y coordinates from meshgrid

P = np.column_stack((X_vector,Y_vector))
Q = f(P[:,0],P[:,1])
Z = f(X,Y)

for ax in [ax1,ax2,ax3,ax4]:

    training_data_size+=4

    training_indices = (len(Q)*scope*sampler.random(n=training_data_size)).astype(int).reshape(training_data_size)
    P_train, Q_train = P[training_indices], Q[training_indices]
    
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=hyperparameter_iterations,alpha=alpha_value)
    gaussian_process.fit(P_train, Q_train)
    print(P_train)
    print(Q_train)
    print(gaussian_process.kernel_)

    mean_prediction, std_prediction = gaussian_process.predict(P, return_std=True)
    
    RMSE = np.sqrt(mean_squared_error(Z.ravel(),mean_prediction))
    
    upper_confidence_interval = mean_prediction + 1.96 * std_prediction
    lower_confidence_interval = mean_prediction - 1.96 * std_prediction
    mean_prediction_grid = mean_prediction.reshape(grid_size,grid_size)
    std_prediction_grid = std_prediction.reshape(grid_size,grid_size)
    upper_confidence_interval_grid = upper_confidence_interval.reshape(grid_size,grid_size)
    lower_confidence_interval_grid = lower_confidence_interval.reshape(grid_size,grid_size)

    lowest_contour = np.amin(mean_prediction)
    highest_contour = np.amax(mean_prediction)
    contour_max_range = highest_contour - lowest_contour

    contour_levels = np.linspace((lowest_contour-0.1*abs(contour_max_range)),(highest_contour+0.1*abs(contour_max_range)),(num_contours+2))

    actual_plot = ax.contour(X, Y, Z, colors='red',levels=contour_levels)
    predicted_plot = ax.contour(X, Y, mean_prediction_grid,colors='blue',levels=contour_levels)

    for contour_level in contour_levels:
        confidence_array = (upper_confidence_interval_grid>=contour_level) & (lower_confidence_interval_grid<=contour_level)
        confidence_plot = ax.contourf(X,Y,confidence_array, levels=[0.5, 2], cmap=ListedColormap(['blue']), alpha=0.2)

    ax.scatter(P_train[:,0],P_train[:,1],marker='x')
    h1,_ = actual_plot.legend_elements()
    h2,_ = predicted_plot.legend_elements()
    h3,_ = confidence_plot.legend_elements()
    ax.legend([h1[0], h2[0], h3[0]], [r"$f(x,y)$", "Mean prediction",r"95% confidence interval"])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    # ax.set_title(f"area sampled from={scope}")
    ax.set_title(f'n={training_data_size}, RMSE={RMSE:.2f}')
    ax.set_aspect('equal','box')
    
fig.suptitle("Gaussian process regression on 2D function")
plt.show()
