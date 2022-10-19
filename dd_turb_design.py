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
import scipy.stats as st
from sklearn.gaussian_process import kernels
from collections import OrderedDict

matern_kernel = kernels.Matern(length_scale = (1,1,1,1,1),
                         length_scale_bounds=(1e-3,1e3),
                         nu=1.5
                         )

constant_kernel_1 = kernels.ConstantKernel(constant_value=1,
                                           constant_value_bounds=(1e-7,1e2)
                                           )

constant_kernel_2 = kernels.ConstantKernel(constant_value=1,
                                           constant_value_bounds=(1e-7,1e2)
                                           )

default_kernel = constant_kernel_1 * matern_kernel + constant_kernel_2

class fit_data:
   def __init__(self,
                training_dataframe,
                kernel_form=default_kernel,
                output_key='eta_lost',
                number_of_restarts=30,
                alpha=0,
                CI_percent=95
                ):
      
      self.number_of_restarts = number_of_restarts
      self.alpha = alpha
      self.output_key = output_key
      self.CI_percent = CI_percent
      self.confidence_scalar = st.norm.ppf(1 - ((1 - (CI_percent / 100)) / 2))
      
      self.input_array_train = training_dataframe.drop(columns=[self.output_key])
      self.output_array_train = training_dataframe[self.output_key]
      
      self.limit_dict = {}
      for column in self.input_array_train:
         self.limit_dict[column] = (np.around(training_dataframe[column].min(),decimals=1),np.around(training_dataframe[column].max(),decimals=1))
      
      gaussian_process = GaussianProcessRegressor(kernel=kernel_form, n_restarts_optimizer=number_of_restarts,alpha=alpha)
      gaussian_process.fit(self.input_array_train, self.output_array_train)
      
      self.optimised_kernel = gaussian_process.kernel_
      self.fitted_function = gaussian_process
      
   def predict(self,
               dataframe,
               include_output=False,
               display_efficiency=True
               ):
      
      if include_output == True:
         self.input_array_test = dataframe.drop(columns=[self.output_key])
         self.output_array_test = dataframe[self.output_key]
      else:
         self.input_array_test = dataframe
      
      self.mean_prediction, self.std_prediction = self.fitted_function.predict(self.input_array_test, return_std=True)
      self.upper_confidence_interval = self.mean_prediction + self.confidence_scalar * self.std_prediction
      self.lower_confidence_interval = self.mean_prediction - self.confidence_scalar * self.std_prediction
      
      if display_efficiency == True:
         self.mean_prediction = (np.ones(len(self.mean_prediction)) - self.mean_prediction)*100
         self.upper_confidence_interval = (np.ones(len(self.upper_confidence_interval)) - self.upper_confidence_interval)*100
         self.lower_confidence_interval = (np.ones(len(self.lower_confidence_interval)) - self.lower_confidence_interval)*100
         if include_output == True:
            self.output_array_test = (np.ones(len(self.output_array_test)) - self.output_array_test)*100
            
      if include_output == True:
         self.RMSE = np.sqrt(mean_squared_error(self.output_array_test,self.mean_prediction))
         
      self.predicted_dataframe = self.input_array_test
      self.predicted_dataframe['output'] = self.mean_prediction
      
      self.min_output = np.amin(self.mean_prediction)
      self.min_output_indices = np.where(self.mean_prediction == self.min_output)
      self.min_output_row = self.predicted_dataframe.iloc[self.min_output_indices]
      
      self.max_output = np.amax(self.mean_prediction)
      self.max_output_indices = np.where(self.mean_prediction == self.max_output)
      self.max_output_row = self.predicted_dataframe.iloc[self.max_output_indices]
      
      return self.mean_prediction,self.upper_confidence_interval,self.lower_confidence_interval
      
   def find_global_max_min_values(self,
                           num_points_interpolate_max=3,
                           limit_dict=None):
         
      if limit_dict == None:
         limit_dict = self.limit_dict
      
      vars_dict = OrderedDict()
      for key in self.limit_dict:
         vars_dict[key] = np.linspace(start=self.limit_dict[key][0], stop=self.limit_dict[key][1], num=num_points_interpolate_max)

      vars_grid_array = np.meshgrid(*vars_dict.values())
      min_max_dataframe = pd.DataFrame({})

      for index,key in enumerate(vars_dict.keys()):
         vars_dict[key] = vars_grid_array[index]
         var_vector = vars_dict[key].ravel()
         min_max_dataframe[key] = var_vector

      self.predict(min_max_dataframe)

      
      self.var_max_dict,self.var_min_dict = {},{}

      for key in vars_dict:
         if len(self.max_output_indices) == 1:
            self.var_max_dict[key] = var_vector[self.max_output_indices[0]]
         else:
            var_max=[]
            for index in self.max_output_indices:
               var_max.append(var_vector[index])
            self.var_max_dict[key] = var_max
            
         if len(self.min_output_indices) == 1:
            self.var_min_dict[key] = var_vector[self.min_output_indices[0]]
         else:
            var_min=[]
            for index in self.min_output_indices:
               var_min.append(var_vector[index])
            self.var_min_dict[key] = var_min
         
      return self.var_max_dict,self.var_min_dict
        
   def plot_vars(self,
                 phi,
                 psi,
                 Lambda,
                 M,
                 Co,
                 limit_dict=None,
                 axis=None,
                 num_points=100,
                 efficiency_step=0.5,
                 opacity=0.3,
                 swap_axis=False,
                 display_efficiency=True,
                 title_variable_spacing=3,
                 plotting_grid_value=[0,0],
                 grid_height=1
                 ):
      
      if axis == None:
         fig,axis = plt.subplots(1,1,sharex=True,sharey=True)
         plot_now = True
      else:
         plot_now = False
      
      var_dict = {'phi':phi,'psi':psi,'Lambda':Lambda,'M':M,'Co':Co}
      
      dimensions = countOf(var_dict.values(), 'vary')
      plot_dataframe = pd.DataFrame({})
      vary_counter = 0
      
      if limit_dict == None:
         limit_dict = self.limit_dict
      
      plot_title = ' '
      
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
            
            if (key == 'M') or (key == 'Co'):
               plot_title += f'{key} = {var_dict[key]}'
               plot_title += '\; '*title_variable_spacing
            else:
               plot_title += '\\' + f'{key} = {var_dict[key]}'
               plot_title += '\; '*title_variable_spacing
      
      if dimensions == 1:

         self.predict(plot_dataframe,display_efficiency=display_efficiency)
         
         if display_efficiency == True:
            axis.set_ylim(bottom=None,top=100,auto=True)
         else:
            axis.set_ylim(bottom=0,top=None,auto=True)
         
         xvar_max = []
         for index in self.max_output_indices:
            xvar_max.append(x1[index])
            axis.text(x1[index], self.mean_prediction[index], f'{self.max_output:.2f}', size=12, color='darkblue')
         
         axis.plot(x1, self.mean_prediction, label=r"Mean prediction", color='blue')
         axis.fill_between(
            x=x1,
            y1=self.upper_confidence_interval,
            y2=self.lower_confidence_interval,
            alpha=opacity,                       
            label=fr"{self.CI_percent}% confidence interval",
            color='orange'
         )
         if plotting_grid_value==[0,0]:
            leg = axis.legend()
            leg.set_draggable(state=True)
         
         if plotting_grid_value[0] == (grid_height-1):
            if (plot_key1 == 'M') or (plot_key1 == 'Co'):
               axis.set_xlabel(fr"${plot_key1}$")
            else:
               xlabel_string = '\\'+plot_key1
               axis.set_xlabel(fr"$ {xlabel_string} $")
               
         if plotting_grid_value[1] == 0:
            axis.set_ylabel('$ \\eta $')
            
         axis.set_xlim(limit_dict[plot_key1][0],limit_dict[plot_key1][1])
   
      elif dimensions == 2:

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
         
         self.predict(plot_dataframe,display_efficiency=display_efficiency)
         
         if display_efficiency == True:
            contour_textlabel = '\\eta'
         else:
            contour_textlabel = '\\eta_{lost}'
         
         min_level = np.round(self.min_output)
         max_level = np.round(self.max_output)
         contour_levels = np.arange(min_level,max_level,efficiency_step)
         
         mean_prediction_grid = self.mean_prediction.reshape(num_points,num_points)
         upper_confidence_interval_grid = self.upper_confidence_interval.reshape(num_points,num_points)
         lower_confidence_interval_grid = self.lower_confidence_interval.reshape(num_points,num_points)

         if swap_axis == False:
            xvar,yvar=X1,X2
         elif swap_axis == True:
            yvar,xvar=X1,X2
            plot_key1,plot_key2=plot_key2,plot_key1
         
         xvar_max,yvar_max=[],[]
         for index in self.max_output_indices:
            xvar_max.append(xvar.ravel()[index])
            yvar_max.append(yvar.ravel()[index])
            axis.text(xvar.ravel()[index], yvar.ravel()[index], f'{self.max_output:.2f}', size=12, color='green')
         
         predicted_plot = axis.contour(xvar, yvar, mean_prediction_grid,levels=contour_levels,cmap='winter',vmin=min_level,vmax=max_level)
         axis.clabel(predicted_plot, inline=1, fontsize=14)
         axis.scatter(xvar_max,yvar_max,color='green',marker='x')

         for contour_level_index,contour_level in enumerate(contour_levels):
            if display_efficiency == True:
               confidence_array = (upper_confidence_interval_grid<=contour_level) & (lower_confidence_interval_grid>=contour_level)
            else:
               confidence_array = (upper_confidence_interval_grid>=contour_level) & (lower_confidence_interval_grid<=contour_level)
            confidence_plot = axis.contourf(xvar,yvar,confidence_array, levels=[0.5, 2], alpha=opacity,cmap = ListedColormap(['orange'])) # cmap=ListedColormap([cm.jet((contour_level-min_level)/(max_level-min_level))]

         if plotting_grid_value==[0,0]:
            h1,_ = predicted_plot.legend_elements()
            h2,_ = confidence_plot.legend_elements()
            leg = axis.legend([h1[0], h2[0]], [fr'$ {contour_textlabel} $, Mean prediction',fr"{self.CI_percent}% confidence interval"])
            leg.set_draggable(state=True)
         
         if plotting_grid_value[0] == (grid_height-1):
            if (plot_key1 == 'M') or (plot_key1 == 'Co'):
               axis.set_xlabel(f"${plot_key1}$")
            else:
               xlabel_string1 = '\\'+plot_key1
               axis.set_xlabel(fr"$ {xlabel_string1} $")
         
         if plotting_grid_value[1] == 0:
            if (plot_key2 == 'M') or (plot_key2 == 'Co'):
               axis.set_ylabel(f"${plot_key2}$")
            else:
               xlabel_string2 = '\\'+plot_key2
               axis.set_ylabel(fr"$ {xlabel_string2} $")
         
         axis.set_xlim(limit_dict[plot_key1][0],limit_dict[plot_key1][1])
         axis.set_ylim(limit_dict[plot_key2][0],limit_dict[plot_key2][1])
          
      else:
         print('INVALID')
         
      axis.set_title(fr'$ {plot_title} $',size=10)
      
      if plot_now == True:
         fig.suptitle("Data-driven turbine design")
         plt.show()
      
   def plot_accuracy(self,
                     testing_dataframe,
                     axis=None
                     ):
      
      self.predict(testing_dataframe,include_output=True)
      
      if axis == None:
         fig,axis = plt.subplots(1,1,sharex=True,sharey=True)
      
      limits_array = np.linspace(self.output_array_test.min(),self.output_array_test.max(),1000)
      axis.scatter(self.output_array_test,self.mean_prediction,marker='x',label='Testing data points')
      axis.plot(limits_array,limits_array,linestyle='dotted',color='red',label = r'$f(x)=x$')
      axis.set_title(fr'RMSE = {self.RMSE:.2e}')
      axis.set_xlabel('$ \\eta $ (actual)')
      axis.set_ylabel('$ \\eta $ (prediction)')
      # axis.set_xlim(limits_array[0],limits_array[-1])
      # axis.set_ylim(limits_array[0],limits_array[-1])
      leg = axis.legend()
      leg.set_draggable(state=True)
      
      if axis == None:
         fig.suptitle("Data-driven turbine design")
         plt.show()
      
   def plot_grid_vars(self,
                      vary_var_1,
                      vary_or_constant_2,
                      column_var,
                      column_var_array,
                      row_var,
                      row_var_array,
                      constant_var,
                      constant_var_value,
                      constant_var_value_2=None,
                      limit_dict=None,
                      num_points=100,
                      efficiency_step=0.5,
                      opacity=0.3,
                      swap_axis=False,
                      display_efficiency=True,
                      title_variable_spacing=3
                      ):
      
      var_dict = {'phi':None,'psi':None,'Lambda':None,'M':None,'Co':None}

      for key in var_dict:
         
         if (key == vary_var_1):
            var_dict[key] = 'vary'
         elif (key == vary_or_constant_2):
            if constant_var_value_2 == None:
               var_dict[key] = 'vary'
            else:
               var_dict[key] = constant_var_value_2
         elif key == column_var:
            var_dict[key] = column_var_array
         elif key == row_var:
            var_dict[key] = row_var_array
         elif key == constant_var:
            var_dict[key] = constant_var_value
      
      num_columns = len(column_var_array)
      num_rows = len(row_var_array)
      
      fig, axes = plt.subplots(nrows=num_rows,
                               ncols=num_columns,
                               sharex=True,
                               sharey=True
                               )

      for (i,j), axis in np.ndenumerate(axes):

         for key in var_dict:
            if column_var == key:
               var_dict[key] = column_var_array[j]
            elif row_var == key:
               var_dict[key] = row_var_array[i]
         
         self.plot_vars(axis=axis,
                        phi=var_dict['phi'],
                        psi=var_dict['psi'],
                        Lambda=var_dict['Lambda'],
                        M=var_dict['M'],
                        Co=var_dict['Co'],
                        num_points=num_points,
                        efficiency_step=efficiency_step,
                        swap_axis=swap_axis,
                        limit_dict=limit_dict,
                        display_efficiency=display_efficiency,
                        title_variable_spacing=title_variable_spacing,
                        opacity=opacity,
                        plotting_grid_value=[i,j],
                        grid_height=num_rows
                        )

      fig.suptitle("Data-driven turbine design")
      
      if (column_var == 'M') or (column_var == 'Co'):
         fig.supxlabel(f"${column_var}$")
      else:
         xlabel_string1 = '\\'+column_var
         fig.supxlabel(fr"$ {xlabel_string1} $")
         
      if (row_var == 'M') or (row_var == 'Co'):
         fig.supylabel(f"${row_var}$")
      else:
         xlabel_string2 = '\\'+row_var
         fig.supylabel(fr"$ {xlabel_string2} $")

      plt.show()