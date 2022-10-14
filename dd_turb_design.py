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
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

class fit_data:
   def __init__(self,kernel_form,training_dataframe,number_of_restarts=30,alpha=0,output_key='eta_lost'):
      
      self.number_of_restarts = number_of_restarts
      self.alpha = alpha
      self.output_key = output_key
      
      self.input_array_train = training_dataframe.drop(columns=[self.output_key])
      self.output_array_train = training_dataframe[self.output_key]
      
      gaussian_process = GaussianProcessRegressor(kernel=kernel_form, n_restarts_optimizer=number_of_restarts,alpha=alpha)
      gaussian_process.fit(self.input_array_train, self.output_array_train)
      
      self.optimised_kernel = gaussian_process.kernel_
      self.fitted_function = gaussian_process
      
   def predict(self,testing_dataframe):
      
      self.input_array_test = testing_dataframe.drop(columns=[self.output_key])
      self.output_array_test = testing_dataframe[self.output_key]
      
      self.mean_prediction, self.std_prediction = self.fitted_function.predict(self.input_array_test, return_std=True)
      self.upper_confidence_interval = self.mean_prediction + 1.96 * self.std_prediction
      self.lower_confidence_interval = self.mean_prediction - 1.96 * self.std_prediction
      
      self.RMSE = np.sqrt(mean_squared_error(self.output_array_test,self.mean_prediction))
      
      return self.mean_prediction
   
   def plot1D(self,phi='vary',psi=0.5,Lambda=0.5,M=0.6,Co=0.5,xmin=0,xmax=2,num_points=20):
      
      var_dict = {'phi':phi,'psi':psi,'Lambda':Lambda,'M':M,'Co':Co}
      plot_dataframe = pd.Dataframe({})
      
      for key in var_dict:
         if var_dict[key] == 'vary':
            plot_dataframe[key] = np.linspace(start=xmin, stop=xmax, num=num_points)
         else:
            plot_dataframe[key] = var_dict[key]*np.ones(num_points)
            
      print(plot_dataframe)
