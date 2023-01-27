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
import sklearn.preprocessing as pre
import compflow_native as compflow

def read_in_data(path='Data Complete',
                 dataset='all'):
   
   dataframe_dict = {}
   for filename in os.listdir(path):
      data_name = str(filename)[:-4]
      
      if dataset=='all':
         pass
      elif dataset=='5D':
         if data_name[:15] != '5D_turbine_data':
            continue
      elif dataset=='4D':
         if data_name[:7] != '4D_data':
            continue
      else:
         if data_name not in dataset:
            continue
      
      filepath = os.path.join(path, filename)
      df = pd.read_csv(filepath)

      df.columns=["phi",
                  "psi", 
                  "Lambda", 
                  "M", 
                  "Co", 
                  "eta_lost",
                  "runid",
                  'Yp_stator', 
                  'Yp_rotor', 
                  'zeta_stator',
                  'zeta_rotor',
                  's_cx_stator',
                  's_cx_rotor',
                  'AR_stator',
                  'AR_rotor',
                  'loss_rat',
                  'Al1',
                  'Al2a',
                  'Al2b',
                  'Al3']
      
      dataframe_dict[data_name] = df

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

def drop_columns(df,variables,output_key):
   for dataframe_variable in df.columns:
      if (dataframe_variable in variables) or (dataframe_variable==output_key):
         pass
      else:
         df=df.drop(columns=str(dataframe_variable))
   return df

class fit_data:  #rename this turb_design and turn init into a new method to fit the data. This will help as model will already be made
   def __init__(self,
                training_dataframe,
                variables=['phi','psi','Lambda','M','Co'],
                output_key='eta_lost',
                number_of_restarts=0,            #do not need to be >0 to optimise parameters. this saves so much time
                length_bounds=(1e-1,1e3),
                noise_magnitude=1e-3,
                noise_bounds=(1e-8,1e-1),
                nu='optimise',
                normalize_y=False,           #seems to make things worse if normalize_y=True
                scale_name='minmax',
                ):
      
      self.number_of_restarts = number_of_restarts
      self.noise_magnitude = noise_magnitude
      self.output_key = output_key
      self.scale_name = scale_name
      self.scale=None
      self.variables = variables
      
      noise_kernel = kernels.WhiteKernel(noise_level=noise_magnitude,
                                         noise_level_bounds=noise_bounds)

      kernel_form = self.matern_kernel(len(variables),bounds=length_bounds) + noise_kernel
      
      if self.scale_name == 'standard':
         self.scale = pre.StandardScaler()
      elif self.scale_name == 'robust':
         self.scale = pre.RobustScaler()
      elif self.scale_name == 'minmax':
         self.scale = pre.MinMaxScaler()
      elif self.scale_name == None:
         pass
      else:
         print('INVALID SCALE NAME')
         quit()
      
      training_dataframe = drop_columns(training_dataframe,variables,output_key)

      if self.scale!=None:
         scaled_dataframe = pd.DataFrame(self.scale.fit_transform(training_dataframe.to_numpy()),
                                           columns=training_dataframe.columns)
         self.input_array_train = scaled_dataframe.drop(columns=[self.output_key])
         self.output_array_train = scaled_dataframe[self.output_key]
      else:
         self.input_array_train = training_dataframe.drop(columns=[self.output_key])
         self.output_array_train = training_dataframe[self.output_key]
         
      self.limit_dict = {}
      for column in self.input_array_train:
         self.limit_dict[column] = (np.around(training_dataframe[column].min(),decimals=1),
                                    np.around(training_dataframe[column].max(),decimals=1)
                                    )
      
      nu_dict = {1.5:None,2.5:None,np.inf:None}
      gaussian_process = GaussianProcessRegressor(kernel=kernel_form,
                                                  n_restarts_optimizer=number_of_restarts,
                                                  normalize_y=normalize_y,
                                                  random_state=0
                                                  )
      if nu=='optimise':
         for nui in nu_dict:
            gaussian_process.set_params(kernel__k1__nu=nui)
            fitted_function = gaussian_process.fit(self.input_array_train.to_numpy(), self.output_array_train.to_numpy())
            nu_dict[nui] = fitted_function.log_marginal_likelihood_value_
         nu = max(nu_dict, key=nu_dict.get)
      
      gaussian_process.set_params(kernel__k1__nu=nu) # kernel__k1__k1__nu if more than 1 kernel (not white), kernel__k1__nu otherwise
      self.fitted_function = gaussian_process.fit(self.input_array_train.to_numpy(), self.output_array_train.to_numpy())
      
      self.optimised_kernel = self.fitted_function.kernel_
      
      if self.scale!=None:
         self.input_array_train = training_dataframe.drop(columns=[self.output_key])
         self.output_array_train = training_dataframe[self.output_key]
      
   def predict(self,
               dataframe,
               include_output=False,
               display_efficiency=True,
               CI_in_dataframe=False,
               CI_percent=95
               ):
      
      dataframe = drop_columns(dataframe,self.variables,self.output_key)
      
      if self.scale!=None:
         if include_output == False:
            dataframe[self.output_key] = np.ones(len(dataframe.index))
            
         scaled_dataframe = pd.DataFrame(self.scale.transform(dataframe.to_numpy()),
                                           columns=dataframe.columns)

         self.input_array_test = scaled_dataframe.drop(columns=[self.output_key])
         self.output_array_test = scaled_dataframe[self.output_key]

      else:
         if include_output == True:
            self.input_array_test = dataframe.drop(columns=[self.output_key])
            self.output_array_test = dataframe[self.output_key]
         else:
            self.input_array_test = dataframe
      
      self.CI_percent = CI_percent
      self.confidence_scalar = st.norm.ppf(1 - ((1 - (CI_percent / 100)) / 2))
      
      if self.scale!=None:
         mean_scaled, std_scaled = self.fitted_function.predict(self.input_array_test.to_numpy(), return_std=True)
         
         if self.scale_name == 'standard':
            self.mean_prediction = mean_scaled*self.scale.scale_[-1] + self.scale.mean_[-1] #derived using probability
            self.std_prediction = self.scale.scale_[-1]*std_scaled #derived using probability
         elif self.scale_name == 'robust':
            self.mean_prediction = mean_scaled * self.scale.scale_[-1] + self.scale.center_[-1]
            self.std_prediction = std_scaled * self.scale.scale_[-1]
            pre.MinMaxScaler()
         elif self.scale_name == 'minmax':
            self.mean_prediction = (mean_scaled - self.scale.min_[-1])/self.scale.scale_[-1]
            self.std_prediction = std_scaled/self.scale.scale_[-1]
         else:
            print('SCALE?')
         if include_output == True:
            self.input_array_test = dataframe.drop(columns=[self.output_key])
            self.output_array_test = dataframe[self.output_key]
         else:
            self.input_array_test = dataframe
            
      else:
         self.mean_prediction, self.std_prediction = self.fitted_function.predict(self.input_array_test.to_numpy(), return_std=True)
      
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
         if self.scale!=None:
            self.score = self.fitted_function.score(scaled_dataframe.drop(columns=[self.output_key]).to_numpy(),scaled_dataframe[self.output_key].to_numpy())
         else:
            self.score = self.fitted_function.score(dataframe.drop(columns=[self.output_key]).to_numpy(),dataframe[self.output_key].to_numpy())
      if CI_in_dataframe == True:
         self.predicted_dataframe['upper'] = self.upper
         self.predicted_dataframe['lower'] = self.lower
         
      
      self.min_output = np.amin(self.mean_prediction)
      self.min_output_indices = np.where(self.mean_prediction == self.min_output)
      
      self.max_output = np.amax(self.mean_prediction)
      self.max_output_indices = np.where(self.mean_prediction == self.max_output)
      
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
                 contour_type='line',
                 plotting_grid=False
                 ):
      
      if axis == None:
         fig,axis = plt.subplots(1,1,sharex=True,sharey=True)
         plot_now = True
      else:
         plot_now = False
         
      if display_efficiency==False:
         efficiency_step = efficiency_step*0.01
      
      color_limits  = np.array([88, 92, 96])
      cmap_colors = ["red","orange","green"]
      
      if display_efficiency == False:
         color_limits = np.flip(1 - (color_limits/100),0)
         cmap_colors = np.flip(cmap_colors)
      
      cmap_norm=plt.Normalize(min(color_limits),max(color_limits))
      cmap_tuples = list(zip(map(cmap_norm,color_limits), cmap_colors))
      efficiency_cmap = mcol.LinearSegmentedColormap.from_list("", cmap_tuples)
      
      plot_dataframe = pd.DataFrame({})
      vary_counter = 0
      
      if limit_dict == None:
         limit_dict = self.limit_dict
      
      plot_title = ' '
      
      var_dict_full = {'phi':phi,'psi':psi,'Lambda':Lambda,'M':M,'Co':Co}
      
      var_dict = {}

      for key, value in var_dict_full.items():
         if key in self.variables:
            var_dict[key] = value
            
      dimensions = countOf(var_dict.values(), 'vary')
               
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
      
      plot_dataframe = drop_columns(plot_dataframe,self.variables,self.output_key)
            
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
         
         if plotting_grid==True:
            y_range = np.amax(self.upper) - np.amin(self.lower)
            axis.set_ylim(bottom=np.amin(self.lower)-0.1*y_range,
                           top=np.amax(self.upper)+0.1*y_range,
                           auto=True)
         else:
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
                           fr"{self.CI_percent}% confidence interval", #edit this
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
      
      if len(var_dict)>2:
         axis.set_title(fr'$ {plot_title} $',size=10)
      
      if plot_now == True:
         fig.tight_layout()
         plt.show()
      
   def plot_accuracy(self,
                     testing_dataframe,
                     axis=None,
                     line_error_percent=5,
                     CI_percent=95,
                     identify_outliers=True,
                     title_variable_spacing=3,
                     display_efficiency=True,
                     plot_errorbars=True
                     ):
      
      runid_dataframe = testing_dataframe['runid']   
      testing_dataframe = drop_columns(testing_dataframe,self.variables,self.output_key)
      
      self.predict(testing_dataframe,
                   include_output=True,
                   CI_in_dataframe=True,
                   CI_percent=CI_percent,
                   display_efficiency=display_efficiency
                   )
      
      if axis == None:
         fig,ax = plt.subplots(1,1,sharex=True,sharey=True)
         
      self.predicted_dataframe['runid'] = runid_dataframe
         
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
               
            ax.scatter(row['actual_output'], row['predicted_output'],color='blue',marker=f'${row_index}$',s=160,label=fr'$ runID={row["runid"]:.0f} $',linewidths=0.1)
      
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
      if plot_errorbars==True:
         ax.errorbar(actual_values,
                     predicted_values,
                     (upper_errorbar,lower_errorbar),
                     fmt='none',
                     capsize=2.0,
                     ecolor='darkblue',
                     label = fr"{self.CI_percent}% confidence interval"
                     )
      ax.set_title(fr'RMSE = {self.RMSE:.2e}    Score: {self.score:.3f}')
      # ax.set_title(fr'Score: {self.score:.3f}')
      
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
                           CI_color=CI_color,
                           plotting_grid=True
                           )
      
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
      
   def matern_kernel(self,
                     N,
                     bounds = (1e-2,1e3)):
      L = np.ones(N)
      L_bounds = []
      for i in range(N):
         L_bounds.append(bounds)
      
      return kernels.Matern(length_scale = L,
                            length_scale_bounds=L_bounds,
                            nu=2.5
                            )
      
   def nondim_to_dim(self,
                     dataframe
                     ):
      
      # Assemble all of the data into the output object
      dataframe['Yp'] = 0.0
      dataframe['Al'] = 0.0
      dataframe['Alrel'] = 0.0
      dataframe['Ma'] = 0.0
      dataframe['Marel'] = 0.0
      dataframe['Ax_Ax1'] = 0.0
      dataframe['Lam'] = 0.0
      dataframe['U_sqrt_cpTo1'] = 0.0
      dataframe['Po_Po1'] = 0.0
      dataframe['To_To1'] = 0.0
      dataframe['Vt_U'] = 0.0
      dataframe['Vtrel_U'] = 0.0
      dataframe['V_U'] = 0.0
      dataframe['Vrel_U'] = 0.0
      dataframe['P3_Po1'] = 0.0
      dataframe['mdot_mdot1'] = 0.0
      
      #   "To1": 1600.0,
      #   "Po1": 1600000.0,
      #   "rgas": 287.14,
      #   "Omega": 314.159,
      #   "delta": 0.1
      #   "htr": 0.9,

      #   "Re": 2000000.0
      
      for index, row in dataframe.iterrows():
         phi = row['phi']                                             # Flow coefficient [--]
         psi = row['psi']                                             # Stage loading coefficient [--]
         Al13 = (row['Al1'],row['Al3'])                               # Yaw angles [deg]
         Ma2 = row['M']                                               # Vane exit Mach number [--]
         ga = 1.33                                                    # Ratio of specific heats [--]
         eta = 1.0 - row['eta_lost']                                  # Polytropic efficiency [--]
         Vx_rat = (row['zeta_stator'],row['zeta_rotor'])              # Axial velocity ratios [--]
         loss_rat = row['loss_rat']                                   # Fraction of stator loss [--]
         
         # Get absolute flow angles using Euler work eqn
         tanAl2 = (np.tan(np.radians(Al13[1])) * Vx_rat[1] + psi / phi)
         Al2 = np.degrees(np.arctan(tanAl2))
         Al = np.insert(Al13, 1, Al2)
         cosAl = np.cos(np.radians(Al))
         
         # Get non-dimensional velocities from definition of flow coefficient
         Vx_U1,Vx_U2,Vx_U3 = Vx_rat[0]*phi, phi, Vx_rat[1]*phi
         Vx_U = np.array([Vx_U1,Vx_U2,Vx_U3])
         Vt_U = Vx_U * np.tan(np.radians(Al))
         V_U = np.sqrt(Vx_U ** 2.0 + Vt_U ** 2.0)

         # Change reference frame for rotor-relative velocities and angles
         Vtrel_U = Vt_U - 1.0
         Vrel_U = np.sqrt(Vx_U ** 2.0 + Vtrel_U ** 2.0)
         Alrel = np.degrees(np.arctan2(Vtrel_U, Vx_U))

         # Use Mach number to get U/cpTo1
         V_sqrtcpTo2 = compflow.V_cpTo_from_Ma(Ma2, ga)
         U_sqrtcpTo1 = V_sqrtcpTo2 / V_U[1]
         Usq_cpTo1 = U_sqrtcpTo1 ** 2.0

         # Non-dimensional temperatures from U/cpTo Ma and stage loading definition
         cpTo1_Usq = 1.0 / Usq_cpTo1
         cpTo2_Usq = cpTo1_Usq
         cpTo3_Usq = (cpTo2_Usq - psi)

         # Turbine
         cpTo_Usq = np.array([cpTo1_Usq, cpTo2_Usq, cpTo3_Usq])
         
         # Mach numbers and capacity from compressible flow relations
         Ma = compflow.Ma_from_V_cpTo(V_U / np.sqrt(cpTo_Usq), ga)
         Marel = Ma * Vrel_U / V_U
         Q = compflow.mcpTo_APo_from_Ma(Ma, ga)
         Q_Q1 = Q / Q[0]

         # Use polytropic effy to get entropy change
         To_To1 = cpTo_Usq / cpTo_Usq[0]
         Ds_cp = -(1.0 - 1.0 / eta) * np.log(To_To1[-1])

         # Somewhat arbitrarily, split loss using loss ratio (default 0.5)
         s_cp = np.hstack((0.0, loss_rat, 1.0)) * Ds_cp

         # Convert to stagnation pressures
         Po_Po1 = np.exp((ga / (ga - 1.0)) * (np.log(To_To1) + s_cp))

         # Account for cooling or bleed flows
         mdot_mdot1 = np.array([1.0, 1.0, 1.0])

         # Use definition of capacity to get flow area ratios
         # Area ratios = span ratios because rm = const
         Dr_Drin = mdot_mdot1 * np.sqrt(To_To1) / Po_Po1 / Q_Q1 * cosAl[0] / cosAl

         # Evaluate some other useful secondary aerodynamic parameters
         T_To1 = To_To1 / compflow.To_T_from_Ma(Ma, ga)
         P_Po1 = Po_Po1 / compflow.Po_P_from_Ma(Ma, ga)
         Porel_Po1 = P_Po1 * compflow.Po_P_from_Ma(Marel, ga)
         
         # Assemble all of the data into the output object
         row['Al1'],row['Al2'],row['Al3'] = Al  #3
         row['Alrel1'],row['Alrel2'],row['Alrel3'] = Alrel  #3
         row['M1'],row['M2'],row['M3'] = Ma  #3
         row['M1rel'],row['M1rel'],row['M1rel'] = Marel  #3
         row['Ax_Ax1'],row['Ax_Ax1'],row['Ax_Ax1'] = Dr_Drin  #3
         row['Po_Po1'],row['Po_Po1'],row['Po_Po1'] = Po_Po1  #3
         row['To_To1'],row['To_To1'],row['To_To1'] = To_To1  #3
         row['Vt_U'],row['Vt_U'],row['Vt_U'] = Vt_U  #3
         row['Vtrel_U'],row['Vtrel_U'],row['Vtrel_U'] = Vtrel_U  #3
         row['V_U'],row['V_U'],row['V_U'] = V_U  #3
         row['Vrel_U'],row['Vrel_U'],row['Vrel_U'] = Vrel_U  #3
         row['P3_Po1'] = P_Po1[2]  #1
         row['mdot_mdot1'],row['mdot_mdot1'],row['mdot_mdot1'] = mdot_mdot1  #3

      return dataframe