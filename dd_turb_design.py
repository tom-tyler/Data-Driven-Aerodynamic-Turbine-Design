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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from operator import countOf
from matplotlib.colors import ListedColormap

class fit_data:
   def __init__(self,kernel_form,training_dataframe,number_of_restarts=30,alpha=0,output_key='eta_lost',confidence_scalar=1.96):
      
      self.number_of_restarts = number_of_restarts
      self.alpha = alpha
      self.output_key = output_key
      self.confidence_scalar = confidence_scalar
      
      self.input_array_train = training_dataframe.drop(columns=[self.output_key])
      self.output_array_train = training_dataframe[self.output_key]
      
      gaussian_process = GaussianProcessRegressor(kernel=kernel_form, n_restarts_optimizer=number_of_restarts,alpha=alpha)
      gaussian_process.fit(self.input_array_train, self.output_array_train)
      
      self.optimised_kernel = gaussian_process.kernel_
      self.fitted_function = gaussian_process
      
   def predict(self,dataframe,include_output=True):
      
      if include_output == True:
         self.input_array_test = dataframe.drop(columns=[self.output_key])
         self.output_array_test = dataframe[self.output_key]
      else:
         self.input_array_test = dataframe
      
      self.mean_prediction, self.std_prediction = self.fitted_function.predict(self.input_array_test, return_std=True)
      self.upper_confidence_interval = self.mean_prediction + self.confidence_scalar * self.std_prediction
      self.lower_confidence_interval = self.mean_prediction - self.confidence_scalar * self.std_prediction
      
      if include_output == True:
         self.RMSE = np.sqrt(mean_squared_error(self.output_array_test,self.mean_prediction))
      
      return self.mean_prediction,self.upper_confidence_interval,self.lower_confidence_interval
   
   def plot_vars(self,axis,phi='vary',psi=0.5,Lambda=0.5,M=0.6,Co=0.5,num_points=100,efficiency_step=0.5,opacity=0.3,swap_axis=False,display_efficiency=True):
      
      var_dict = {'phi':phi,'psi':psi,'Lambda':Lambda,'M':M,'Co':Co}
      limit_dict = {'phi':(0,1.5),'psi':(0,3),'Lambda':(0,1),'M':(0,0.85),'Co':(0,1)}
      
      dimensions = countOf(var_dict.values(), 'vary')
      plot_dataframe = pd.DataFrame({})
      vary_counter = 0
      
      for key in var_dict:
         if (var_dict[key] == 'vary') and (vary_counter == 0):
            plot_key1 = key
            x1 = np.linspace(start=limit_dict[key][0], stop=limit_dict[key][1], num=num_points)
            plot_dataframe[key] = x1
            vary_counter += 1
         elif (var_dict[key] == 'vary') and (vary_counter == 1):
            plot_key2 = key
            x2 = np.linspace(start=limit_dict[key][0], stop=limit_dict[key][1], num=num_points)
            plot_dataframe[key] = x2
         else:
            plot_dataframe[key] = var_dict[key]*np.ones(num_points)
      
      if dimensions == 1:

         self.predict(plot_dataframe,include_output=False)
         
         if display_efficiency == True:
            self.mean_prediction = (np.ones(len(self.mean_prediction)) - self.mean_prediction)*100
            self.upper_confidence_interval = (np.ones(len(self.upper_confidence_interval)) - self.upper_confidence_interval)*100
            self.lower_confidence_interval = (np.ones(len(self.lower_confidence_interval)) - self.lower_confidence_interval)*100
            axis.set_ylim(bottom=None,top=100,auto=True)
         else:
            self.mean_prediction = (self.mean_prediction)*100
            self.upper_confidence_interval = (self.upper_confidence_interval)*100
            self.lower_confidence_interval = (self.lower_confidence_interval)*100
            axis.set_ylim(bottom=0,top=None,auto=True)
         
         axis.plot(x1, self.mean_prediction, label=r"Mean prediction", color='blue')
         axis.fill_between(
            x=x1,
            y1=self.upper_confidence_interval,
            y2=self.lower_confidence_interval,
            alpha=opacity,                       
            label=r"95% confidence interval",
            color='blue'
         )
         axis.legend()
         if (plot_key1 == 'M') or (plot_key1 == 'Co'):
            axis.set_xlabel(f"${plot_key1}$")
         else:
            axis.set_xlabel(f"${plot_key1}$")
         axis.set_ylabel('$eta$')
         axis.set_xlim(limit_dict[plot_key1][0],limit_dict[plot_key1][1])
   
      elif dimensions == 2:

         # x1_grid = np.linspace(start=limit_dict[plot_key1][0], stop=limit_dict[plot_key1][1], num=num_points)
         # x2_grid = np.linspace(start=limit_dict[plot_key2][0], stop=limit_dict[plot_key2][1], num=num_points)
         X1,X2 = np.meshgrid(x1,x2) # creates two matrices which vary across in x and y
         X1_vector = X1.ravel() #vector of "all" x coordinates from meshgrid
         X2_vector = X2.ravel() #vector of "all" y coordinates from meshgrid
         plot_dataframe = pd.DataFrame({})
         for key in var_dict:
            if key == plot_key1:
               plot_dataframe[key] = X1_vector
               vary_counter += 1
            elif key == plot_key2:
               plot_dataframe[key] = X2_vector
            else:
               plot_dataframe[key] = var_dict[key]*np.ones(num_points**2)
         
         self.predict(plot_dataframe,include_output=False)
         
         if display_efficiency == True:
            self.mean_prediction = (np.ones(len(self.mean_prediction)) - self.mean_prediction)*100
            self.upper_confidence_interval = (np.ones(len(self.upper_confidence_interval)) - self.upper_confidence_interval)*100
            self.lower_confidence_interval = (np.ones(len(self.lower_confidence_interval)) - self.lower_confidence_interval)*100
         else:
            self.mean_prediction = (self.mean_prediction)*100
            self.upper_confidence_interval = (self.upper_confidence_interval)*100
            self.lower_confidence_interval = (self.lower_confidence_interval)*100
         
         min_level = np.round(np.amin(self.mean_prediction))
         max_eta = np.amax(self.mean_prediction)
         max_eta_indices = np.where(self.mean_prediction == max_eta)
         max_level = np.round(max_eta)
         contour_levels = np.arange(min_level,max_level,efficiency_step)
         
         mean_prediction_grid = self.mean_prediction.reshape(num_points,num_points)
         upper_confidence_interval_grid = self.upper_confidence_interval.reshape(num_points,num_points)
         lower_confidence_interval_grid = self.lower_confidence_interval.reshape(num_points,num_points)

         if swap_axis == False:
            xvar,yvar=X1,X2
         elif swap_axis == True:
            yvar,xvar=X1,X2
         
         xvar_max,yvar_max=[],[]
         for index in max_eta_indices:
            xvar_max.append(xvar.ravel()[index])
            yvar_max.append(yvar.ravel()[index])
            axis.text(xvar.ravel()[index], yvar.ravel()[index], f'{max_eta:.2f}', size=12, color='green')
         
         predicted_plot = axis.contour(xvar, yvar, mean_prediction_grid,levels=contour_levels,cmap='winter',vmin=min_level,vmax=max_level)
         axis.clabel(predicted_plot, inline=1, fontsize=14)
         axis.scatter(xvar_max,yvar_max,color='green',marker='x')

         for contour_level_index,contour_level in enumerate(contour_levels):
            if display_efficiency == True:
               confidence_array = (upper_confidence_interval_grid<=contour_level) & (lower_confidence_interval_grid>=contour_level)
            else:
               confidence_array = (upper_confidence_interval_grid>=contour_level) & (lower_confidence_interval_grid<=contour_level)
            confidence_plot = axis.contourf(xvar,yvar,confidence_array, levels=[0.5, 2], alpha=opacity,cmap = ListedColormap(['orange'])) # cmap=ListedColormap([cm.jet((contour_level-min_level)/(max_level-min_level))]

         h1,_ = predicted_plot.legend_elements()
         h2,_ = confidence_plot.legend_elements()
         axis.legend([h1[0], h2[0]], ['$eta$',r"95% confidence interval"])
         
         if (plot_key1 == 'M') or (plot_key1 == 'Co'):
            axis.set_xlabel(f"${plot_key1}$")
         else:
            axis.set_xlabel(f"${plot_key1}$")
            
         if (plot_key2 == 'M') or (plot_key2 == 'Co'):
            axis.set_ylabel(f"${plot_key2}$")
         else:
            axis.set_ylabel(f"${plot_key2}$")
         
         axis.set_xlim(limit_dict[plot_key1][0],limit_dict[plot_key1][1])
         axis.set_ylim(limit_dict[plot_key2][0],limit_dict[plot_key2][1])
         
      else:
         print('INVALID')
