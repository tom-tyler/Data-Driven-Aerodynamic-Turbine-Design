"""goal is to create a ML model for a 2 input function (hyperbolic paraboloid)"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from matplotlib.colors import ListedColormap
from scipy.interpolate import UnivariateSpline

grid_size=1000
x = np.linspace(start=-10, stop=10, num=grid_size)
y = np.linspace(start=-10, stop=10, num=grid_size)
# print(np.dstack((x,y)))
X,Y = np.meshgrid(x,y) # creates two matrices which vary across in x and y
X_vector = X.ravel() #vector of "all" x coordinates from meshgrid
Y_vector = Y.ravel() #vector of "all" y coordinates from meshgrid
P = np.column_stack((X_vector,Y_vector))
Q = P[:,0]**2 - P[:,1]**2
Z = X**2 - Y**2

training_indices = np.random.randint(low=len(Q),size=10)
P_train, Q_train = P[training_indices], Q[training_indices]

kernel = 1.0 * RBF(length_scale=(1.0,1.0))
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gaussian_process.fit(P_train, Q_train)
print(gaussian_process.kernel_)

testing_indices = np.random.randint(low=len(Q),size=20)
P_train = P[testing_indices]

mean_prediction, std_prediction = gaussian_process.predict(P, return_std=True)
upper_confidence_interval = mean_prediction + 1.96 * std_prediction
lower_confidence_interval = mean_prediction - 1.96 * std_prediction
mean_prediction_grid = mean_prediction.reshape(grid_size,grid_size)
std_prediction_grid = std_prediction.reshape(grid_size,grid_size)
upper_confidence_interval_grid = upper_confidence_interval.reshape(grid_size,grid_size)
lower_confidence_interval_grid = lower_confidence_interval.reshape(grid_size,grid_size)

contour_levels = np.linspace(-80,80,5)
#if row upper_grid

fig, ax = plt.subplots()
actual_plot = ax.contour(X, Y, Z, colors='red',levels=contour_levels)
predicted_plot = ax.contour(X, Y, mean_prediction_grid,colors='blue',levels=contour_levels)
confidence_plot_upper = ax.contour(X, Y, upper_confidence_interval_grid,levels=contour_levels,colors='green')
confidence_plot_lower = ax.contour(X, Y, lower_confidence_interval_grid,levels=contour_levels,colors='green')

for contour_level in contour_levels:
    confidence_array = (upper_confidence_interval_grid>=contour_level) & (lower_confidence_interval_grid<=contour_level)
    ax.contourf(X,Y,confidence_array, levels=[0.5, 2], cmap=ListedColormap(['green']), alpha=0.3)



# for contour_level in range(len(contour_levels)):
    
#     contour_lines_upper = confidence_plot_upper.allsegs[contour_level]
#     contour_lines_lower = confidence_plot_lower.allsegs[contour_level]
#     #plot when between level and std
    
#     for contour_line in range(len(contour_lines_upper)):
        
#         upper_contour_x = contour_lines_upper[contour_line][:,1]

        
#         lower_contour_x = contour_lines_lower[contour_line][:,1]
        
        
        

# ax.scatter(P_train[:,0],P_train[:,1],marker='x')
# (zzmax <= 1) & (zzmin >= 1)
h1,_ = actual_plot.legend_elements()
h2,_ = predicted_plot.legend_elements()
ax.legend([h1[0], h2[0]], [r"$f(x,y) = x^{2}-y^{2}$", "Mean prediction"])
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_title("Gaussian process regression on multivariate function")
ax.set_aspect('equal','box')
plt.show()
# actual_plot = plt.contour(X,Y,Z, label=r"$f(x,y) = x^{2}-y^{2}$", colors='red')

# plt.plot(P, Q, label=r"$f(x,y) = x^{2}-y^{2}$", linestyle="dotted")
# plt.scatter(P_train, Q_train, label=r"$Samples$", marker="x")
# predicted_plot = plt.contour(X,Y, mean_prediction_grid, label="Mean prediction",colors='blue')
# plt.fill_between(
#     x=X.ravel(),
#     y1=upper_confidence_interval,   # 95% of area is within 1.96x standard deviation of the mean
#     y2=lower_confidence_interval,
#     alpha=0.5,                                       #transparency
#     label=r"95% confidence interval"
# )
