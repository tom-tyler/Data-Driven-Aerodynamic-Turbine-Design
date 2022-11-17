from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from operator import countOf
import scipy.stats as st
from sklearn.gaussian_process import kernels
from collections import OrderedDict
import os
import matplotlib.colors as mcol

matern_kernel = kernels.Matern(length_scale = (1,1,1,1,1),
                         length_scale_bounds=((1e-2,1e2),(1e-2,1e2),(1e-2,1e2),(1e-2,1e2),(1e-2,1e2)),
                         nu=2.5
                         )

constant_kernel_1 = kernels.ConstantKernel(constant_value=1,
                                           constant_value_bounds=(1e-7,1e2)
                                           )

constant_kernel_2 = kernels.ConstantKernel(constant_value=1,
                                           constant_value_bounds=(1e-7,1e2)
                                           )

noise_kernel = kernels.WhiteKernel(noise_level=1e-6,
                                   noise_level_bounds=(1e-10,1e-4))

default_kernel = matern_kernel + noise_kernel

def fix_vars(df,
             vars_to_fix,
             values
             ):
   if vars_to_fix==None:
      return df
   else:
      for i, var in enumerate(vars_to_fix):
         if values=='mean':
            df[var] = np.mean(df[var])*np.ones(df.shape[0])
         else:
            df[var] = values[i]*np.ones(df.shape[0])
      
   return df #do not need to take return value

def read_in_data(path='Data',
                 dataset='all'
                 ):
   
   dataframe_dict = {}
   for filename in os.listdir(path):
      data_name = str(filename)[:-4]
      
      if dataset=='all':
         pass
      elif dataset=='5D':
         if data_name[:-2] != '5D_turbine_data':
            continue
      else:
         if data_name not in dataset:
            continue
      
      filepath = os.path.join(path, filename)
      df = pd.read_csv(filepath)
   
      if len(df.columns)==7:
         df.columns=["phi", "psi", "Lambda", "M", "Co", "eta_lost","runid"]
         df = df.drop(columns=['runid'])
         dataframe_dict[data_name] = df
         
      elif len(df.columns)==6: #back-compatibility
         df.columns=["phi", "psi", "Lambda", "M", "Co", "eta_lost"]
         dataframe_dict[data_name] = df
         
      else:
         print('error, invalid csv')
         quit()

   dataframe_list = dataframe_dict.values()   
   data = pd.concat(dataframe_list,ignore_index=True)
   return data

def split_data(df,
               fraction_training=0.75,
               random_seed_state=2
               ):
   training_data = df.sample(frac=fraction_training,
                             random_state=random_seed_state
                             )
   testing_data = df.loc[~df.index.isin(training_data.index)]
   return training_data,testing_data

class fit_data:
   def __init__(self,
                training_dataframe,
                kernel_form=default_kernel,
                output_key='eta_lost',
                number_of_restarts=30,
                noise_magnitude=0,
                nu='optimise'
                ):
      
      self.number_of_restarts = number_of_restarts
      self.noise_magnitude = noise_magnitude
      self.output_key = output_key

      self.input_array_train = training_dataframe.drop(columns=[self.output_key])
      self.output_array_train = training_dataframe[self.output_key]
      
      self.limit_dict = {}
      for column in self.input_array_train:
         self.limit_dict[column] = (np.around(training_dataframe[column].min(),decimals=1),np.around(training_dataframe[column].max(),decimals=1))
      
      nu_dict = {1.5:None,2.5:None,np.inf:None}
      gaussian_process = GaussianProcessRegressor(kernel=kernel_form, n_restarts_optimizer=number_of_restarts,alpha=noise_magnitude)
      if nu=='optimise':
         for nui in nu_dict:
            gaussian_process.set_params(kernel__k1__nu=nui)
            fitted_function = gaussian_process.fit(self.input_array_train, self.output_array_train)
            nu_dict[nui] = fitted_function.log_marginal_likelihood_value_
         nu = max(nu_dict, key=nu_dict.get)
      
      gaussian_process.set_params(kernel__k1__nu=nu) # kernel__k1__k1__nu if more than 1 kernel (not white), kernel__k1__nu otherwise
      self.fitted_function = gaussian_process.fit(self.input_array_train, self.output_array_train)
      
      self.optimised_kernel = self.fitted_function.kernel_
      
   def predict(self,
               dataframe,
               include_output=False,
               display_efficiency=True,
               CI_in_dataframe=False,
               CI_percent=95
               ):
      
      if include_output == True:
         self.input_array_test = dataframe.drop(columns=[self.output_key])
         self.output_array_test = dataframe[self.output_key]
      else:
         self.input_array_test = dataframe
      
      self.CI_percent = CI_percent
      self.confidence_scalar = st.norm.ppf(1 - ((1 - (CI_percent / 100)) / 2))
      
      self.mean_prediction, self.std_prediction = self.fitted_function.predict(self.input_array_test, return_std=True)
      self.upper = self.mean_prediction + self.confidence_scalar * self.std_prediction
      self.lower = self.mean_prediction - self.confidence_scalar * self.std_prediction
      
      if display_efficiency == True:
         self.mean_prediction = (np.ones(len(self.mean_prediction)) - self.mean_prediction)*100
         self.lower, self.upper = (np.ones(len(self.upper)) - self.upper)*100, (np.ones(len(self.lower)) - self.lower)*100
         if include_output == True:
            self.output_array_test = (np.ones(len(self.output_array_test)) - self.output_array_test)*100
         self.training_output = (np.ones(len(self.output_array_train)) - self.output_array_train)*100
      else:
         self.training_output = self.output_array_train
            
      self.predicted_dataframe = self.input_array_test
      self.predicted_dataframe['predicted_output'] = self.mean_prediction
      if include_output == True:
         self.RMSE = np.sqrt(mean_squared_error(self.output_array_test,self.mean_prediction))
         self.predicted_dataframe['actual_output'] = self.output_array_test
         self.predicted_dataframe['percent_error'] = abs((self.mean_prediction - self.output_array_test)/self.output_array_test)*100
         self.score = self.fitted_function.score(dataframe.drop(columns=[self.output_key]),dataframe[self.output_key])
      if CI_in_dataframe == True:
         self.predicted_dataframe['upper'] = self.upper
         self.predicted_dataframe['lower'] = self.lower
         
      
      self.min_output = np.amin(self.mean_prediction)
      self.min_output_indices = np.where(self.mean_prediction == self.min_output)
      
      self.max_output = np.amax(self.mean_prediction)
      self.max_output_indices = np.where(self.mean_prediction == self.max_output)
      
      print(self.predicted_dataframe)
      
      return self.predicted_dataframe
      
   def find_global_max_min_values(self,
                                  num_points_interpolate=20,
                                  limit_dict=None):
         
      if limit_dict != None:
         self.limit_dict = limit_dict
      
      vars_dict = OrderedDict()
      for key in self.limit_dict:
         vars_dict[key] = np.linspace(start=self.limit_dict[key][0], stop=self.limit_dict[key][1], num=num_points_interpolate)

      vars_grid_array = np.meshgrid(*vars_dict.values())
      min_max_dataframe = pd.DataFrame({})

      for index,key in enumerate(vars_dict.keys()):
         vars_dict[key] = vars_grid_array[index]
         var_vector = vars_dict[key].ravel()
         min_max_dataframe[key] = var_vector

      min_max_dataframe = self.predict(min_max_dataframe)
      
      self.min_output_row = min_max_dataframe.iloc[self.min_output_indices]
      self.max_output_row = min_max_dataframe.iloc[self.max_output_indices]
         
      return self.max_output_row,self.min_output_row
        
   def plot_vars(self,
                 phi='mean',
                 psi='mean',
                 Lambda='mean',
                 M='mean',
                 Co='mean',
                 limit_dict=None,
                 axis=None,
                 num_points=100,
                 efficiency_step=0.5,
                 opacity=0.2,
                 swap_axis=False,
                 display_efficiency=True,
                 title_variable_spacing=3,
                 plotting_grid_value=[0,0],
                 grid_height=1,
                 CI_percent=95,
                 plot_training_points=False,
                 legend_outside=False,
                 CI_color='orange',
                 contour_type='line'
                 ):
      
      if axis == None:
         fig,axis = plt.subplots(1,1,sharex=True,sharey=True)
         plot_now = True
      else:
         plot_now = False
         
      if display_efficiency==False:
         efficiency_step = efficiency_step*0.01
      
      var_dict = {'phi':phi,'psi':psi,'Lambda':Lambda,'M':M,'Co':Co}
      
      color_limits  = np.array([88, 92, 96])
      cmap_colors = ["red","orange","green"]
      
      if display_efficiency == False:
         color_limits = np.flip(1 - (color_limits/100),0)
         cmap_colors = np.flip(cmap_colors)
      
      cmap_norm=plt.Normalize(min(color_limits),max(color_limits))
      cmap_tuples = list(zip(map(cmap_norm,color_limits), cmap_colors))
      efficiency_cmap = mcol.LinearSegmentedColormap.from_list("", cmap_tuples)
      
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
            if (var_dict[key] == 'mean'):
               plot_dataframe[key] = np.mean(self.input_array_train[key])*np.ones(num_points)
               constant_value = np.mean(self.input_array_train[key])
            else:
               plot_dataframe[key] = var_dict[key]*np.ones(num_points)
               constant_value = var_dict[key]
            
            if (key == 'M') or (key == 'Co'):
               plot_title += f'{key} = {constant_value:.3f}'
               plot_title += '\; '*title_variable_spacing
            else:
               plot_title += '\\' + f'{key} = {constant_value:.3f}'
               plot_title += '\; '*title_variable_spacing
      
      if dimensions == 1:

         self.predict(plot_dataframe,
                      display_efficiency=display_efficiency,
                      CI_percent=CI_percent)
         
         if display_efficiency == True:
            xvar_max = []
            for index in self.max_output_indices:
               xvar_max.append(x1[index])
               axis.text(x1[index], self.mean_prediction[index], f'{self.max_output:.2f}', size=12, color='darkblue')
         else:
            xvar_min = []
            for index in self.min_output_indices:
               xvar_min.append(x1[index])
               axis.text(x1[index], self.mean_prediction[index], f'{self.min_output:.2f}', size=12, color='darkblue')
         
         if plot_training_points == True:
            axis.scatter(x=self.input_array_train[plot_key1],
                         y=self.training_output,
                         marker='x',
                         color='red',
                         label='Training data points')
         
         
         axis.plot(x1, self.mean_prediction, label=r"Mean prediction", color='blue')
         axis.fill_between(
            x=x1,
            y1=self.upper,
            y2=self.lower,
            alpha=opacity,                       
            label=fr"{self.CI_percent}% confidence interval",
            color='blue'
         )
         if plotting_grid_value==[0,0]:
            if legend_outside == True:
               leg = axis.legend(loc='upper left',
                                 bbox_to_anchor=(1.02,1.0),
                                 borderaxespad=0,
                                 frameon=True,
                                 ncol=1,
                                 prop={'size': 10})
            else:
               leg = axis.legend()
            leg.set_draggable(state=True)
         
         if plotting_grid_value[0] == (grid_height-1):
            if (plot_key1 == 'M') or (plot_key1 == 'Co'):
               axis.set_xlabel(fr"${plot_key1}$")
            else:
               xlabel_string = '\\'+plot_key1
               axis.set_xlabel(fr"$ {xlabel_string} $")
               
         if plotting_grid_value[1] == 0:
            if display_efficiency == True:
               axis.set_ylabel('$ \\eta $')
            else:
               axis.set_ylabel('$ \\eta_{lost} $')
            
         axis.set_xlim(limit_dict[plot_key1][0],limit_dict[plot_key1][1],
                       auto=True)
         
         # if display_efficiency == True:
         #    y_range = np.amax(self.lower) - np.amin(self.upper)
         #    axis.set_ylim(top=np.amax(self.lower)+0.1*y_range,
         #                  bottom=np.amin(self.upper)-0.1*y_range,
         #                  auto=True)
         # else:
         y_range = np.amax(self.upper) - np.amin(self.lower)
         axis.set_ylim(bottom=np.amin(self.lower)-0.1*y_range,
                        top=np.amax(self.upper)+0.1*y_range,
                        auto=True)

         axis.grid(linestyle = '--', linewidth = 0.5)
         
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
               if (var_dict[key] == 'mean'):
                  plot_dataframe[key] = np.mean(self.input_array_train[key])*np.ones(num_points**2)
               else:
                  plot_dataframe[key] = var_dict[key]*np.ones(num_points**2)
            
         self.predict(plot_dataframe,
                      display_efficiency=display_efficiency,
                      CI_percent=CI_percent)

         
         if display_efficiency == True:
            contour_textlabel = '\\eta'
         else:
            contour_textlabel = '\\eta_{lost}'
         
         min_level = np.floor(self.min_output/efficiency_step)*efficiency_step
         max_level = np.ceil(self.max_output/efficiency_step)*efficiency_step
         contour_levels = np.arange(min_level,max_level,efficiency_step)
         
         mean_prediction_grid = self.mean_prediction.reshape(num_points,num_points)
         upper_grid = self.upper.reshape(num_points,num_points)
         lower_grid = self.lower.reshape(num_points,num_points)

         if swap_axis == False:
            xvar,yvar=X1,X2
         elif swap_axis == True:
            yvar,xvar=X1,X2
            plot_key1,plot_key2=plot_key2,plot_key1
         
         xvar_max,yvar_max=[],[]
         for index in self.max_output_indices:
            xvar_max.append(xvar.ravel()[index])
            yvar_max.append(yvar.ravel()[index])
            axis.text(xvar.ravel()[index], yvar.ravel()[index], f'{self.max_output:.2f}', size=12, color='darkgreen')
         
         if plot_training_points == True:
            training_points_plot = axis.scatter(x=self.input_array_train[plot_key1],
                         y=self.input_array_train[plot_key2],
                         marker='x',
                         color='blue'
                         )
         
         if contour_type=='line':
            predicted_plot = axis.contour(xvar, yvar, mean_prediction_grid,levels=contour_levels,cmap=efficiency_cmap,norm=cmap_norm)
            axis.clabel(predicted_plot, inline=1, fontsize=14)
            for contour_level_index,contour_level in enumerate(contour_levels):
               # if display_efficiency == True:
               #    confidence_array = (upper_grid<=contour_level) & (lower_grid>=contour_level)
               # else:
               confidence_array = (upper_grid>=contour_level) & (lower_grid<=contour_level)

               contour_color = efficiency_cmap(cmap_norm(contour_level))

               confidence_plot = axis.contourf(xvar,yvar,confidence_array, levels=[0.5, 2], alpha=opacity,cmap = mcol.ListedColormap([contour_color])) 
               h2,_ = confidence_plot.legend_elements()
               
         elif contour_type=='continuous':
            predicted_plot = axis.contourf(xvar, yvar, mean_prediction_grid,cmap=efficiency_cmap,norm=cmap_norm,levels=contour_levels,extend='both')
            
         h1,_ = predicted_plot.legend_elements()
         axis.scatter(xvar_max,yvar_max,color='green',marker='x')

         if plotting_grid_value==[0,0]:
            
            if plot_training_points == True:
               if contour_type == 'line':
                  handles = [h1[0], h2[0], training_points_plot]
                  labels = [fr'$ {contour_textlabel} $, Mean prediction',
                           fr"{self.CI_percent}% confidence interval",
                           'Training data points']
               else:
                  handles = [h1[0], training_points_plot]
                  labels = [fr'$ {contour_textlabel} $, Mean prediction',
                            'Training data points']
            else:
               if contour_type == 'line':
                  handles = [h1[0], h2[0]]
                  labels = [fr'$ {contour_textlabel} $, Mean prediction',
                           fr"{self.CI_percent}% confidence interval",
                           'Training data points']
               else:
                  handles = [h1[0]]
                  labels = [fr'$ {contour_textlabel} $, Mean prediction']
            if legend_outside == True:
               leg = axis.legend(handles=handles,
                                 labels=labels,
                                 loc='upper left',
                                 bbox_to_anchor=(1.02,1.0),
                                 borderaxespad=0,
                                 frameon=True,
                                 ncol=1,
                                 prop={'size': 10})
            else:
               leg = axis.legend(handles=handles,
                                 labels=labels)

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
         
         axis.set_xlim(limit_dict[plot_key1][0],
                       limit_dict[plot_key1][1],
                       auto=True)
         axis.set_ylim(limit_dict[plot_key2][0],
                       limit_dict[plot_key2][1],
                       auto=True)
         
         axis.grid(linestyle = '--', linewidth = 0.5)
          
      else:
         print('INVALID')
      
      axis.set_title(fr'$ {plot_title} $',size=10)
      
      if plot_now == True:
         fig.suptitle("Data-driven turbine design")
         fig.tight_layout()
         plt.show()
      
   def plot_accuracy(self,
                     testing_dataframe,
                     axis=None,
                     line_error_percent=5,
                     CI_percent=95,
                     identify_outliers=True,
                     title_variable_spacing=3,
                     display_efficiency=True
                     ):
      
      self.predict(testing_dataframe,
                   include_output=True,
                   CI_in_dataframe=True,
                   CI_percent=CI_percent,
                   display_efficiency=display_efficiency
                   )
      
      if axis == None:
         fig,ax = plt.subplots(1,1,sharex=True,sharey=True)
         
      predicted_values = self.predicted_dataframe['predicted_output']
      actual_values = self.predicted_dataframe['actual_output']
      upper_errorbar = (self.predicted_dataframe['upper']-predicted_values)
      lower_errorbar = (predicted_values-self.predicted_dataframe['lower'])
      
      if identify_outliers == True:
         outliers = self.predicted_dataframe[self.predicted_dataframe['percent_error'] > line_error_percent]
               
         for row_index,row in outliers.iterrows():
            value_string = f''
            newline=' $\n$ '
            for col_index,col in enumerate(outliers):
               
               if (col == 'M') or (col == 'Co'):
                  if (col_index%2==0) and (col_index!=0):
                     value_string += newline
                  value_string += f'{col}={row[col]:.3f}'
                  value_string += '\; '*title_variable_spacing
                  
               elif (col == 'phi') or (col == 'psi') or (col == 'Lambda'):
                  if (col_index%2==0) and (col_index!=0):
                     value_string += newline
                  value_string += '\\' + f'{col}={row[col]:.3f}'
                  value_string += '\; '*title_variable_spacing
               
            ax.scatter(row['actual_output'], row['predicted_output'],color='blue',marker=f'${row_index}$',s=160,label=fr'$ {value_string} $',linewidths=0.1)
      
      limits_array = np.linspace(actual_values.min(),actual_values.max(),1000)
      upper_limits_array = (1+line_error_percent/100)*limits_array
      lower_limits_array = (1-line_error_percent/100)*limits_array
      
      if identify_outliers == True:
         non_outliers = self.predicted_dataframe[self.predicted_dataframe['percent_error'] < line_error_percent]
         ax.scatter(non_outliers['actual_output'],non_outliers['predicted_output'],marker='x',label='Testing data points',color='blue')
      else:
         ax.scatter(actual_values,predicted_values,marker='x',label='Test data points',color='blue')
      ax.plot(limits_array,limits_array,linestyle='solid',color='red',label = r'$f(x)=x$')
      ax.plot(limits_array,upper_limits_array,linestyle='dotted',color='red',label = f'{line_error_percent}% error interval')
      ax.plot(limits_array,lower_limits_array,linestyle='dotted',color='red')
      ax.errorbar(actual_values,
                  predicted_values,
                  (upper_errorbar,lower_errorbar),
                  fmt='none',
                  capsize=2.0,
                  ecolor='darkblue',
                  label = fr"{self.CI_percent}% confidence interval"
                  )
      ax.set_title(fr'RMSE = {self.RMSE:.2e}    Score: {self.score:.3f}')
      if display_efficiency== True:
         ax.set_xlabel('$ \\eta $ (actual)')
         ax.set_ylabel('$ \\eta $ (prediction)')
      else:
         ax.set_xlabel('$ \\eta_{lost} $ (actual)')
         ax.set_ylabel('$ \\eta_{lost} $ (prediction)')
         
      leg = ax.legend(loc='upper left',
                      bbox_to_anchor=(1.02,1.0),
                      borderaxespad=0,
                      frameon=True,
                      ncol=1,
                      prop={'size': 10})
      leg.set_draggable(state=True)
      
      ax.grid(linestyle = '--', linewidth = 0.5)
      
      if axis == None:
         fig.suptitle("Data-driven turbine design")
         fig.tight_layout()
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
                      title_variable_spacing=3,
                      with_arrows=True,
                      CI_percent=95,
                      plot_training_points=False,
                      legend_outside=False,
                      CI_color='orange'
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
      
      if num_columns == 1:
         for i, axis in enumerate(axes):

            for key in var_dict:
               if column_var == key:
                  var_dict[key] = column_var_array[0]
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
                           plotting_grid_value=[i,0],
                           grid_height=num_rows,
                           CI_percent=CI_percent,
                           plot_training_points=plot_training_points,
                           legend_outside=legend_outside,
                           CI_color=CI_color
                           )
      elif num_rows==1:
         for j, axis in enumerate(axes):

            for key in var_dict:
               if column_var == key:
                  var_dict[key] = column_var_array[j]
               elif row_var == key:
                  var_dict[key] = row_var_array[0]
            
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
                           plotting_grid_value=[0,j],
                           grid_height=num_rows,
                           CI_percent=CI_percent,
                           plot_training_points=plot_training_points,
                           legend_outside=legend_outside,
                           CI_color=CI_color
                           )
      
      else:
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
                           grid_height=num_rows,
                           CI_percent=CI_percent,
                           plot_training_points=plot_training_points,
                           legend_outside=legend_outside,
                           CI_color=CI_color
                           )

      fig.suptitle("Data-driven turbine design")
      
      if with_arrows==True:
         if (column_var == 'M') or (column_var == 'Co'):
            fig.supxlabel(f"${column_var} \\rightarrow $")
         else:
            xlabel_string1 = '\\'+column_var+' \\rightarrow'
            fig.supxlabel(fr"$ {xlabel_string1} $")
            
         if (row_var == 'M') or (row_var == 'Co'):
            fig.supylabel(f"$\\leftarrow {row_var} $")
         else:
            xlabel_string2 = '\\leftarrow \\'+row_var
            fig.supylabel(fr"$ {xlabel_string2} $")
      else:
         if (column_var == 'M') or (column_var == 'Co'):
            fig.supxlabel(f"${column_var} $")
         else:
            xlabel_string1 = '\\'+column_var
            fig.supxlabel(fr"$ {xlabel_string1} $")
            
         if (row_var == 'M') or (row_var == 'Co'):
            fig.supylabel(f"${row_var} $")
         else:
            xlabel_string2 = '\\'+row_var
            fig.supylabel(fr"$ {xlabel_string2} $")

      
      plt.show()