"""
General Method:

1.) Have n data points of m-dimensional data

2.) If not already in this form, put into the form of a
(nx1) array, with each element of the array a (1xm) array:

3.) Create a kernel:
- consider length scales
- consider potential noise
- consider type of data
- use previous runs to inform choice of constant and lengths potentially

4.) Fit training data using kernel

5.) Predict outcome of testing data and compare to actual results.
Calculate the RMSE and plot predicted vs actual

6.) Ability to plot graphs holding all but one or 2 values constant

"""
"""
Functions Required:

1.) Fit data:
 - inputs:
    - kernel_form
    - num restarts
    - alpha(noise)
    - training matrix:
        [[x1,y1,z1],
         [x2,y2,z2],
         ...]
         
 - attributes:
    - output the kernel
    - output gaussian_process (this is fitted function, so rename as such)

 - methods:
    - 

2.) Predict result (use now fitted function):
 - inputs:
    - fitted gaussian process
    - testing matrix without output:
        [[x1,y1],
         [x2,y2],
         ...]
         
 - attributes:
    - mean_prediction
    - std_prediction

 - methods:
    - output RMSE (array [z1,z2,z3,...].T)
    - confidence intervals (% confidence)
        - upper_confidence_interval = mean_prediction + number * std_prediction
        - lower_confidence_interval = mean_prediction - number * std_prediction
 
3.) plot graphs
 - inputs:
    - x1
    - x2 (if you want a 2d plot, else none)
    - y
"""

from sklearn.gaussian_process import GaussianProcessRegressor

class fit_data:
   def __init__(self,kernel_form,training_dataframe,number_of_restarts=30,alpha=0,output_key='eta_lost'):
      
      self.number_of_restarts = number_of_restarts
      self.alpha = alpha
      
      self.input_array = training_dataframe.drop(columns=[output_key])
      self.output_array = training_dataframe[output_key]
      
      gaussian_process = GaussianProcessRegressor(kernel=kernel_form, n_restarts_optimizer=number_of_restarts,alpha=alpha)
      gaussian_process.fit(self.input_array, self.output_array)
      
      self.optimised_kernel = gaussian_process.kernel_
      self.fitted_function = gaussian_process