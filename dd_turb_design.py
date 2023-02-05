from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.metrics import mean_squared_error
import sklearn.preprocessing as pre
import numpy as np
import pandas as pd
import scipy.stats as st
import scipy.optimize as sciop
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
from collections import OrderedDict
import compflow_native as compflow
import os
import sys

def read_in_data(dataset='4D',
                 path='Data Complete',
                 factor=5,
                 state_retention_statistics=False,
                 ignore_incomplete=False
                 ):
   
   dataframe_dict = {}
   n_before = 0
   n_after = 0
   for filename in os.listdir(path):
      data_name = str(filename)[:-4]
      
      if dataset=='all':
         pass
      elif dataset=='5D only':
         if data_name[:15] != '5D_turbine_data':
            continue
      elif dataset=='4D only':
         if (data_name[:7] != '4D_data'):
            continue
      elif dataset=='2D only':
         if data_name[:15] != '2D_phi_psi_data':
            continue
      elif dataset in ['2D','4D','5D']:
         pass
      else:
         if data_name not in dataset:
            continue
      
      filepath = os.path.join(path, filename)
      df = pd.read_csv(filepath)

      if df.shape[1] == 20:
         df.columns=["phi",
                     "psi", 
                     "Lambda", 
                     "M2", 
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
         
      elif df.shape[1] == 6:
         df.columns=["phi",
                  "psi", 
                  "Lambda", 
                  "M2", 
                  "Co", 
                  "eta_lost"]
         df['runid'] = 0
         df['Yp_stator'] = 0 
         df['Yp_rotor'] = 0 
         df['zeta_stator'] = 0
         df['zeta_rotor'] = 0
         df['s_cx_stator'] = 0
         df['s_cx_rotor'] = 0
         df['AR_stator'] = 0
         df['AR_rotor'] = 0
         df['loss_rat'] = 0
         df['Al1'] = 0
         df['Al2a'] = 0
         df['Al2b'] = 0
         df['Al3'] = 0
         if ignore_incomplete==True:
            continue
            
      elif df.shape[1] == 7:
         df.columns=["phi",
                     "psi", 
                     "Lambda", 
                     "M2", 
                     "Co", 
                     "eta_lost",
                     "runid"]
         df['Yp_stator'] = 0 
         df['Yp_rotor'] = 0 
         df['zeta_stator'] = 0
         df['zeta_rotor'] = 0
         df['s_cx_stator'] = 0
         df['s_cx_rotor'] = 0
         df['AR_stator'] = 0
         df['AR_rotor'] = 0
         df['loss_rat'] = 0
         df['Al1'] = 0
         df['Al2a'] = 0
         df['Al2b'] = 0
         df['Al3'] = 0
         if ignore_incomplete==True:
            continue
      else:
         sys.exit('Invalid dataframe')
      
      # filter by factor% error
      lower_factor = 1 - factor/100
      upper_factor = 1 + factor/100
      
      n_before += len(df.index)
      
      if dataset in ['4D','4D only']:
         # Lambda = 0.5
         val = 0.5
         df = df[df["Lambda"] < upper_factor*val]
         df = df[df["Lambda"] > lower_factor*val]
         
      elif dataset in ['2D','2D only']:
         # Lambda = 0.5
         val=0.5
         df = df[df["Lambda"] < upper_factor*val]
         df = df[df["Lambda"] > lower_factor*val]
         # M2 = 0.7 or 0.65
         val_h=0.7
         val_l=0.65
         df = df[df["M2"] < upper_factor*val_h]
         df = df[df["M2"] > lower_factor*val_l]
         # Co = 0.65 or 0.7
         val_h=0.7
         val_l=0.65
         df = df[df["Co"] < upper_factor*val_h]
         df = df[df["Co"] > lower_factor*val_l]
         
      df = df.reindex(sorted(df.columns), axis=1)
         
      n_after += len(df.index)
      
      dataframe_dict[data_name] = df

   if state_retention_statistics==True:
      print(f'n_before = {n_before}\nn_after = {n_after}\n%retained = {n_after/n_before*100:.2f} %')
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
                variables=None,
                output_key='eta_lost',
                number_of_restarts=0,            #do not need to be >0 to optimise parameters. this saves so much time
                length_bounds=(1e-1,1e3),
                noise_magnitude=1e-3,
                noise_bounds=(1e-8,1e-1),
                nu='optimise',
                normalize_y=False,           #seems to make things worse if normalize_y=True
                scale_name=None,
                extra_variable_options=False,
                iterate_extra_params=False
                ):
      
      if variables==None:
         sys.exit('Please state variable to fit over.')
      elif output_key==None:
         sys.exit('Please state output to fit to.')
      
      self.number_of_restarts = number_of_restarts
      self.noise_magnitude = noise_magnitude
      self.output_key = output_key
      self.scale_name = scale_name
      self.scale=None
      self.variables = variables
      self.fit_dimensions = len(self.variables)
      self.no_points = len(training_dataframe.index)
      # print('traindf\n',training_dataframe.head())
      
      
      
      noise_kernel = kernels.WhiteKernel(noise_level=noise_magnitude,
                                         noise_level_bounds=noise_bounds)

      kernel_form = self.matern_kernel(len(variables),bounds=length_bounds) + noise_kernel
      
      if extra_variable_options==True:
         training_dataframe = extra_nondim_params(training_dataframe,iterate=iterate_extra_params)
         self.scale_name=None
         
         # Currently, this is overcomplicated as it iterates over 
         # alpha to be able to use Lambda as an input. However, we already have 
         # alpha in large initially dataset, so this could be improved.
         
         # This version of the function is not useless though, as 'trained' the '
         # predicted' dataset does not have alpha 3 value in it, so maybe this could be used later.
         # However, need to think about which values have been fixed etc before being able to do this 
         # as constants are not the same any more
      
      if self.scale_name == 'standard':
         self.scale = pre.StandardScaler()
         scales = [self.scale]
      elif self.scale_name == 'robust':
         self.scale = pre.RobustScaler()
         scales = [self.scale]
      elif self.scale_name == 'minmax':
         self.scale = pre.MinMaxScaler()
         scales = [self.scale]
      elif self.scale_name == None:
         pass
         scales = [None]
      elif self.scale_name == 'optimise':
         scales=[pre.StandardScaler(),pre.RobustScaler(),pre.MinMaxScaler()]
         scale_name_list = ['standard','robust','minmax']
         maxLMLV=0
      else:
         sys.exit('INVALID SCALE NAME')
      
      training_dataframe = drop_columns(training_dataframe,variables,output_key)
      training_dataframe = training_dataframe.reindex(sorted(training_dataframe.columns), axis=1)
      # print('traindf\n',training_dataframe.head())
      
      for i,scale_type in enumerate(scales):
      
         if scale_type!=None:
            scaled_dataframe = pd.DataFrame(scale_type.fit_transform(training_dataframe.to_numpy()),
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
         
         if self.scale_name == 'optimise':
            LMLV = fitted_function.log_marginal_likelihood_value_
            # print('scale_type',scale_type,LMLV)
            if LMLV > maxLMLV:
               maxLMLV=LMLV
               optimum_scale_type = scale_type
               opt_scale_i = i
            
         self.optimised_kernel = self.fitted_function.kernel_
         
         if scale_type!=None:
            self.input_array_train = training_dataframe.drop(columns=[self.output_key])
            self.output_array_train = training_dataframe[self.output_key]
      
      if self.scale_name == 'optimise':
         self.scale = optimum_scale_type
         self.scale_name = scale_name_list[opt_scale_i]
         
      self.min_train_output = np.min([self.output_array_train])
      self.max_train_output = np.max([self.output_array_train])
      
   def predict(self,
               dataframe,
               include_output=False,
               display_efficiency=True,
               CI_in_dataframe=False,
               CI_percent=95
               ):
      # print('predictdf\n',dataframe.head())
      dataframe = dataframe.reindex(sorted(dataframe.columns), axis=1)
      # print('predictdf\n',dataframe.head())
      dataframe = drop_columns(dataframe,self.variables,self.output_key)
      # print('predictdf\n',dataframe.head())
      
      if self.scale!=None:
         if include_output == False:
            dataframe[self.output_key] = np.ones(len(dataframe.index)) #this is a 'dummy' output for 
                                                                       #the data scaling, as it was fitted with 5 inputs.
                                                                       # Is removed after
            
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
         elif self.scale_name == 'minmax':
            self.mean_prediction = (mean_scaled - self.scale.min_[-1])/self.scale.scale_[-1]
            self.std_prediction = std_scaled/self.scale.scale_[-1]
         else:
            sys.exit('Incorrect scale?')
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
                 x1=None,
                 x2=None,
                 constants='mean',
                 limit_dict=None,
                 axis=None,
                 num_points=100,
                 efficiency_step=0.5,
                 opacity=0.2,
                 display_efficiency=True,
                 title_variable_spacing=3,
                 plotting_grid_value=[0,0],
                 grid_height=1,
                 CI_percent=95,
                 plot_training_points=False,
                 legend_outside=False,
                 contour_type='line',
                 show_max=True,
                 show_min=False,
                 state_no_points=False,
                 plot_actual_data=False,
                 plot_actual_data_filter_factor=5,
                 show_actual_with_model=True
                 ):
      
      if axis == None:
         fig,axis = plt.subplots(1,1,sharex=True,sharey=True)
         plot_now = True
      else:
         plot_now = False
      
      
      
      # color_limits  = np.array([(100-100*self.max_train_output)*0.9,
      #                           np.mean([(100-100*self.min_train_output),(100-100*self.max_train_output)]),
      #                           (100-100*self.min_train_output)*1.1])
      # print(color_limits)
      color_limits = [88,92,96]
      cmap_colors = ["red","orange","green"]
      
      if display_efficiency == False:
         color_limits = np.flip(1 - (color_limits/100),0)
         cmap_colors = np.flip(cmap_colors)
         efficiency_step = efficiency_step*0.01
         show_max=False
         show_min=True
         contour_textlabel = '\\eta_{lost}'
      else:
         contour_textlabel = '\\eta'
            
      
      cmap_norm=plt.Normalize(min(color_limits),max(color_limits))
      cmap_tuples = list(zip(map(cmap_norm,color_limits), cmap_colors))
      efficiency_cmap = mcol.LinearSegmentedColormap.from_list("", cmap_tuples)
      
      plot_dataframe = pd.DataFrame({})
      
      if limit_dict == None:
         limit_dict = self.limit_dict
      
      plot_title = ' '

      constants_check=self.variables.copy()
                    
      if (x1 != None) and (x2 == None):
         plot_key1 = x1
         plot_dataframe[plot_key1] = np.linspace(start=limit_dict[plot_key1][0], stop=limit_dict[plot_key1][1], num=num_points)
         constants_check.remove(plot_key1)  
         dimensions=1   
      elif (x1 == None) and (x2 != None):
         plot_key1 = x2
         plot_dataframe[plot_key1] = np.linspace(start=limit_dict[plot_key1][0], stop=limit_dict[plot_key1][1], num=num_points)
         constants_check.remove(plot_key1)
         dimensions=1
      elif (x1 != None) and (x2 != None):
         plot_key1 = x1
         plot_dataframe[plot_key1] = np.linspace(start=limit_dict[plot_key1][0], stop=limit_dict[plot_key1][1], num=num_points)
         constants_check.remove(plot_key1)
         plot_key2 = x2
         plot_dataframe[plot_key2] = np.linspace(start=limit_dict[plot_key2][0], stop=limit_dict[plot_key2][1], num=num_points)
         constants_check.remove(plot_key2)
         dimensions=2
      else:
         sys.exit("Please specify x or y") 
      
      constant_value = {}
      
      if constants == 'mean':
         for constant_key in constants_check:
            constant_value[constant_key] = np.mean(self.input_array_train[constant_key])
         
      elif set(constants_check) != set(constants):
         sys.exit("Constants specified are incorrect")
         
      else:
         # format of constants is {'M2':0.7,'Co':0.6, ...}
         for constant_key in constants:
            if (constants[constant_key] == 'mean'):
               constant_value[constant_key] = np.mean(self.input_array_train[constant_key])
            else:
               constant_value[constant_key] = constants[constant_key]
               
      for constant_key in constants_check:
         if constant_key in ['phi','psi','Lambda']:
            plot_title += '\\' + f'{constant_key} = {constant_value[constant_key]:.3f}'
            plot_title += '\; '*title_variable_spacing
         else:
            plot_title += f'{constant_key} = {constant_value[constant_key]:.3f}'
            plot_title += '\; '*title_variable_spacing
      
      if dimensions == 2:

         X1,X2 = np.meshgrid(plot_dataframe[plot_key1],
                             plot_dataframe[plot_key2]) # creates two matrices which vary across in x and y
         X1_vector = X1.ravel() #vector of "all" x coordinates from meshgrid
         X2_vector = X2.ravel() #vector of "all" y coordinates from meshgrid
         plot_dataframe = pd.DataFrame({})
         plot_dataframe[plot_key1] = X1_vector
         plot_dataframe[plot_key2] = X2_vector
      
      for constant_key in constants_check:
         plot_dataframe[constant_key] = constant_value[constant_key]*np.ones(num_points**dimensions)

      self.predict(plot_dataframe,
                   display_efficiency=display_efficiency,
                   CI_percent=CI_percent)
            
      if plot_actual_data == True:
            # filter by factor% error
            lower_factor = 1 - plot_actual_data_filter_factor/100
            upper_factor = 1 + plot_actual_data_filter_factor/100
            actual_data_df = pd.concat([self.input_array_train.copy(),self.output_array_train.copy()],axis=1)
            
            for constant_key in constants_check:
               val = constant_value[constant_key]
               actual_data_df = actual_data_df[actual_data_df[constant_key] < upper_factor*val]
               actual_data_df = actual_data_df[actual_data_df[constant_key] > lower_factor*val]

            if display_efficiency==True:
               actual_data_df[self.output_key] = (1 - actual_data_df[self.output_key])*100
            
      if dimensions == 1:
         
         if plot_training_points == True:
            axis.scatter(x=self.input_array_train[plot_key1],
                         y=self.training_output,
                         marker='x',
                         color='red',
                         label='Training data points')

         if show_max == True:
            max_i = np.squeeze(self.max_output_indices)
            axis.text(plot_dataframe[plot_key1][max_i], self.mean_prediction[max_i], f'{self.max_output:.2f}', size=12, color='darkblue')

         if show_min == True:
            min_i = np.squeeze(self.min_output_indices)
            axis.text(plot_dataframe[plot_key1][min_i], self.mean_prediction[min_i], f'{self.min_output:.2f}', size=12, color='darkblue')

         if plot_actual_data==True:
               
            poly_degree = int(0.75*actual_data_df.shape[0])
            if poly_degree > 3:
               poly_degree = 3
               
            coefs = np.polynomial.polynomial.polyfit(x=actual_data_df[plot_key1],
                                                     y=actual_data_df[self.output_key],
                                                     deg=poly_degree)

            fit_function = np.polynomial.polynomial.Polynomial(coefs)    # instead of np.poly1d

            x_actual_fit = np.linspace(np.min(actual_data_df[plot_key1]),np.max(actual_data_df[plot_key1]),50)
            y_actual_fit = fit_function(x_actual_fit)
               
            axis.scatter(actual_data_df[plot_key1],
                      actual_data_df[self.output_key],
                      color='darkorange',
                      marker='x')
            axis.plot(x_actual_fit,
                      y_actual_fit,
                      label=r'Polynomial curve from actual data',
                      color='orange',
                      zorder=1e3)
            
         if show_actual_with_model == True:
            
            axis.plot(plot_dataframe[plot_key1], 
                      self.mean_prediction, 
                      label=r'Mean prediction', 
                      color='blue'
                      )
            
            axis.fill_between(x=plot_dataframe[plot_key1],
                              y1=self.upper,
                              y2=self.lower,
                              alpha=opacity,                       
                              label=fr"{self.CI_percent}% confidence interval",
                              color='blue'
                              )
            
            y_range = np.amax(self.upper) - np.amin(self.lower)
            axis.set_xlim(limit_dict[plot_key1][0],
                        limit_dict[plot_key1][1],
                        auto=True)
            axis.set_ylim(bottom=np.amin(self.lower)-0.1*y_range,
                           top=np.amax(self.upper)+0.1*y_range,
                           auto=True)
            
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
            if plot_key1 in ['phi','psi','Lambda']:
               xlabel_string = '\\'+plot_key1
               axis.set_xlabel(fr"$ {xlabel_string} $")
            else:
               axis.set_xlabel(fr"${plot_key1}$")
               
         if plotting_grid_value[1] == 0:
            if display_efficiency == True:
               axis.set_ylabel('$ \\eta $')
            else:
               axis.set_ylabel('$ \\eta_{lost} $')

         axis.grid(linestyle = '--', linewidth = 0.5)
         
      elif dimensions == 2:
         
         min_level = np.floor(self.min_output/efficiency_step)*efficiency_step
         max_level = np.ceil(self.max_output/efficiency_step)*efficiency_step
         contour_levels = np.arange(min_level,max_level,efficiency_step)
         
         mean_prediction_grid = self.mean_prediction.reshape(num_points,num_points)
         upper_grid = self.upper.reshape(num_points,num_points)
         lower_grid = self.lower.reshape(num_points,num_points)
         
         if show_max == True:
            max_i = np.squeeze(self.max_output_indices)
            axis.text(X1.ravel()[max_i], X2.ravel()[max_i], f'{self.max_output:.2f}', size=12, color='dark'+cmap_colors[2])
            axis.scatter(X1.ravel()[max_i], X2.ravel()[max_i],color=cmap_colors[2],marker='x')

         if show_min == True:
            min_i = np.squeeze(self.min_output_indices)
            axis.text(X1.ravel()[min_i], X2.ravel()[min_i], f'{self.min_output:.2f}', size=12, color='dark'+cmap_colors[0])
            axis.scatter(X1.ravel()[min_i], X2.ravel()[min_i],color=cmap_colors[0],marker='x')
         
         if plot_training_points == True:
            training_points_plot = axis.scatter(x=self.input_array_train[plot_key1],
                                                y=self.input_array_train[plot_key2],
                                                marker='x',
                                                color='blue'
                                                )
         
         if contour_type=='line':
            predicted_plot = axis.contour(X1, X2, mean_prediction_grid,levels=contour_levels,cmap=efficiency_cmap,norm=cmap_norm)
            axis.clabel(predicted_plot, inline=1, fontsize=14)
            for contour_level_index,contour_level in enumerate(contour_levels):  #clear this up

               confidence_array = (upper_grid>=contour_level) & (lower_grid<=contour_level)

               contour_color = efficiency_cmap(cmap_norm(contour_level))

               confidence_plot = axis.contourf(X1,X2,confidence_array, levels=[0.5, 2], alpha=opacity,cmap = mcol.ListedColormap([contour_color])) 
               h2,_ = confidence_plot.legend_elements()
               
         elif contour_type=='continuous':
            predicted_plot = axis.contourf(X1, X2, mean_prediction_grid,cmap=efficiency_cmap,norm=cmap_norm,levels=contour_levels,extend='both')
         
         else:
            sys.exit('Please specify "continuous" or "line" for contour_type')
            
         h1,_ = predicted_plot.legend_elements()
         
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
            if plot_key1 in ['phi','psi','Lambda']:
               xlabel_string1 = '\\'+plot_key1
               axis.set_xlabel(fr"$ {xlabel_string1} $")
            else:
               axis.set_xlabel(f"${plot_key1}$")
         
         if plotting_grid_value[1] == 0:
            if plot_key2 in ['phi','psi','Lambda']:
               xlabel_string2 = '\\'+plot_key2
               axis.set_ylabel(fr"$ {xlabel_string2} $")
            else:
               axis.set_ylabel(f"${plot_key2}$")
         
         axis.set_xlim(limit_dict[plot_key1][0],
                       limit_dict[plot_key1][1],
                       auto=True)
         axis.set_ylim(limit_dict[plot_key2][0],
                       limit_dict[plot_key2][1],
                       auto=True)
         
         axis.grid(linestyle = '--', linewidth = 0.5)
          
      else:
         sys.exit('Somehow wrong number of dimensions')
      
      if self.fit_dimensions>2:
         axis.set_title(fr'$ {plot_title} $',size=10)
      
      if plot_now == True:
         fig.tight_layout()
         plt.show()
         
      return plot_dataframe
      
   def plot_accuracy(self,
                     testing_dataframe,
                     axis=None,
                     line_error_percent=5,
                     CI_percent=95,
                     identify_outliers=True,
                     title_variable_spacing=3,
                     display_efficiency=True,
                     plot_errorbars=True,
                     score_variable='both'
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
               
               if col in ['phi','psi','Lambda']:
                  if (col_index%2==0) and (col_index!=0):
                     value_string += newline
                  value_string += '\\' + f'{col}={row[col]:.3f}'
                  value_string += '\; '*title_variable_spacing
               else:
                  if (col_index%2==0) and (col_index!=0):
                     value_string += newline
                  value_string += f'{col}={row[col]:.3f}'
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
      if score_variable=='both':
         ax.set_title(fr'RMSE = {self.RMSE:.2e}    $R^2$ = {self.score:.3f}')
      elif score_variable=='R2':
         ax.set_title(fr'$R^2$ = {self.score:.3f}')
      elif score_variable=='RMSE':
         ax.set_title(fr'RMSE = {self.RMSE:.2e}')
      else:
         sys.exit("Enter suitable score variable from ['both','R2','RMSE']")
      
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
      
   def plot(self,
            x1=None,
            x2=None,
            constants='mean',          # form: {'M':0.5,'Co':0.5}
            gridvars={},               # form: {'M':[0.5,0.6,0.7],'Co:[0.6,0.7]}
            rotate_grid=False,
            limit_dict=None,
            num_points=100,
            efficiency_step=0.5,
            opacity=0.3,
            display_efficiency=True, 
            title_variable_spacing=3,
            with_arrows=True,
            CI_percent=95,
            plot_training_points=False,
            legend_outside=False,
            contour_type='line',
            show_max=True,
            show_min=False,
            state_no_points=False,
            plot_actual_data=False,
            plot_actual_data_filter_factor=5,
            show_actual_with_model=True
            ):

      grid_constants=self.variables.copy()
      
      if x1 != None:
         grid_constants.remove(x1)  
      if x2 != None:
         grid_constants.remove(x2)  
      if (x1==None) and (x2==None):
         sys.exit('Must state wither correct x1 or correct x2')
      
      grid_shape, grid_keys=[1,1], {0:' ',1:' '}
      
      if rotate_grid==True:
         grid_index=1
      else:
         grid_index=0
         
      if gridvars != {}:
         for var in grid_constants:
            if var in [x1,x2]:
               sys.exit('Already plotting grid variable')
            elif var in gridvars:
               grid_shape[grid_index] = len(gridvars[var])
               grid_keys[grid_index] = var
               grid_index = not grid_index
      else:
         grid_index=0
               
      num_rows=grid_shape[0]
      num_columns=grid_shape[1]
      
      fig, axes = plt.subplots(nrows=num_rows,
                               ncols=num_columns,
                               sharex=True,
                               sharey=True
                               )
      
      for indices, axis in np.ndenumerate(axes):
         
         if (num_columns == 1) and (num_rows > 1):
            i = np.squeeze(indices)
            j = 0
         elif (num_columns > 1) and (num_rows == 1):
            j = np.squeeze(indices)
            i = 0
         elif (num_columns > 1) and (num_rows > 1):
            (i,j) = indices
         else:
            i,j=0,0
            
         constant_dict = {}
         
         for var in grid_constants:
            if (var in gridvars) and (grid_keys[0]==var):
               constant_dict[var] = gridvars[var][i]

            elif (var in gridvars) and (grid_keys[1]==var):
               constant_dict[var] = gridvars[var][j]
            else:
               if constants=='mean':
                  constant_dict[var] = 'mean'
               else:
                  constant_dict[var] = constants[var]
         self.plot_vars(x1=x1,
                        x2=x2,
                        constants=constant_dict,
                        limit_dict=limit_dict,
                        axis=axis,
                        num_points=num_points,
                        efficiency_step=efficiency_step,
                        opacity=opacity,
                        display_efficiency=display_efficiency,
                        title_variable_spacing=title_variable_spacing,
                        plotting_grid_value=[i,j],
                        grid_height=num_rows,
                        CI_percent=CI_percent,
                        plot_training_points=plot_training_points,
                        legend_outside=legend_outside,
                        contour_type=contour_type,
                        show_max=show_max,
                        show_min=show_min,
                        state_no_points=state_no_points,
                        plot_actual_data=plot_actual_data,
                        plot_actual_data_filter_factor=plot_actual_data_filter_factor,
                        show_actual_with_model=show_actual_with_model
                        )

      if (num_columns>1) or (num_rows>1):
         if with_arrows==True:
            if num_columns >1:
               if grid_keys[1] in ['phi','psi','Lambda']:
                  xlabel_string1 = '\\'+grid_keys[1]+' \\rightarrow'
                  fig.supxlabel(fr"$ {xlabel_string1} $")
               else:
                  fig.supxlabel(f"$ {grid_keys[1]} \\rightarrow $")
            if num_rows >1:
               if grid_keys[0] in ['phi','psi','Lambda']:
                  xlabel_string2 = '\\leftarrow \\'+grid_keys[0]
                  fig.supylabel(fr"$ {xlabel_string2} $")
               else:
                  fig.supylabel(f"$\\leftarrow {grid_keys[0]} $")
         else:
            if num_columns >1:
               if grid_keys[1] in ['phi','psi','Lambda']:
                  xlabel_string1 = '\\'+grid_keys[1]
                  fig.supxlabel(fr"$ {xlabel_string1} $")
               else:
                  fig.supxlabel(f"${grid_keys[1]} $")
            if num_rows >1:   
               if grid_keys[0] in ['phi','psi','Lambda']:
                  xlabel_string2 = '\\'+grid_keys[0]
                  fig.supylabel(fr"$ {xlabel_string2} $")
               else:
                  fig.supylabel(f"${grid_keys[0]} $")
                  
      if state_no_points==True:
         fig.suptitle(f'n = {self.no_points}')
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

def dim_2_nondim(shaft_power=25e6,
                  stagnation_pressure_ratio=1.5,
                  blade_number=40,
                  turbine_diameter=1.6,
                  mdot=275,
                  T01=1600,
                  p01=1600000,
                  shaft_speed=100*np.pi,
                  aspect_ratio=1.6,
                  hub_to_tip_ratio=0.9):
   
   gamma = 1.33
   R = 272.9
   cp = R / (1 - 1/gamma)
   # print('cp=',cp)
   tip_radius = 0.5*turbine_diameter
   hub_radius = hub_to_tip_ratio*tip_radius
   mean_radius = (tip_radius+hub_radius)/2
   span = tip_radius-hub_radius
   chord = span/aspect_ratio
   
   A = np.pi*2*mean_radius*span
   
   T02 = T01
   T03 = T01 - shaft_power/(mdot*cp)
   h01 = cp*T01
   h02 = cp*T02
   h03 = cp*T03
   
   U = mean_radius*shaft_speed
   
   

   # print('U=',U)

   pitch = mean_radius*2*np.pi/blade_number
   dh0 = h03 - h01
   
   mcpT01_Ap01 = mdot*np.sqrt(cp*T01) / (A * p01)
   # print('mcpT01_Ap01=',mcpT01_Ap01)
   if mcpT01_Ap01 > 1.28:
      sys.exit('Too large a mass flow function - flow will choke first')
   
   M1 = compflow.to_Ma("mcpTo_APo",mcpT01_Ap01,gamma)
   p1 = p01 / compflow.from_Ma('Po_P',M1,gamma)
   T1 = T01 / compflow.from_Ma('To_T',M1,gamma)
   
   p03 = p01/stagnation_pressure_ratio
   mcpT03_Ap03 = mdot*np.sqrt(cp*T03) / (A * p03)
   # print('T03=',T03)
   # print('A=',A)
   # print('mcpT03_Ap03=',mcpT03_Ap03)
   if mcpT03_Ap03 > 1.28:
      sys.exit('Too large a mass flow function - flow will choke first')
   
   M3 = compflow.to_Ma("mcpTo_APo",mcpT03_Ap03,gamma)
   T3 = T03 / compflow.from_Ma('To_T',M3,gamma)
   
   #assume axial stator inflow, and constant Vx
   V1 = M1*np.sqrt(gamma*R*T1)
   Vx = V1
   Vt1 = 0
   V3 = M3*np.sqrt(gamma*R*T3)
   # print('M3=',M3)
   # print('T3',T3)
   
   Vt3 = np.sqrt(V3**2-Vx**2)
   Vt2 = Vt3 - dh0/U
   V2 = np.sqrt(Vt2**2+Vx**2)
   
   h1 = h01 - 0.5*V1**2
   h2 = h02 - 0.5*V2**2
   h3 = h03 - 0.5*V3**2
   # print('Vt1,Vt2,Vt3 = ',Vt1,Vt2,Vt3)
   # print('V1,V2,V3 = ',V1,V2,V3)
   # print('h01,h02,h03 = ',h01,h02,h03)
   # print('h1,h2,h3 = ',h1,h2,h3)
   T2 = h2/cp
   # print('T2=',T2)
   M2 = np.squeeze(compflow.to_Ma("To_T",T02/T2,gamma))
   # print(M2)
   phi = Vx/U
   psi = -1*dh0/U**2
   Lambda = np.abs((h3-h2)/(h3-h1))
   
   a1=0
   a2=np.arctan(Vt2/Vx)
   a3=np.arctan(Vt3/Vx)
   
   # chord below should be replaced by suction surface length
   Co = [(pitch/chord)*(np.tan(a1)-np.tan(a2))*np.cos(a2), (pitch/chord)*(np.tan(a2)-np.tan(a3))*np.cos(a3)]
   
   return [phi,psi,Lambda,M2,Co]
   
def extra_nondim_params(dataframe, iterate=False):
      
   def vars_from_Al(x,index,df,iterating=False):

      if iterating==True:
         Al13 = (0.0,x)                            
         loss_ratio=0.4
         zeta_stator=1
         zeta_rotor=1
      else:                          
         loss_ratio=df.loc[index,'loss_rat']  
         zeta_stator=df.loc[index,'zeta_stator']  
         zeta_rotor=df.loc[index,'zeta_rotor']  
      
      phi = df.loc[index,'phi']                
      psi = df.loc[index,'psi']          
      Ma2 = df.loc[index,'M2']                  
      ga = 1.33                          
      
      if iterating==True:
         # Get absolute flow angles using Euler work eqn
         tanAl2 = (np.tan(np.radians(Al13[1]))*zeta_rotor + psi / phi)
         Al2 = np.degrees(np.arctan(tanAl2))
         Al = np.insert(Al13, 1, Al2)
      else:
         Al2 = np.mean([df.loc[index,'Al2a'],df.loc[index,'Al2b']])
         Al = np.array([df.loc[index,'Al1'], Al2, df.loc[index,'Al3']])
         
      cosAl = np.cos(np.radians(Al))
      
      # Get non-dimensional velocities from definition of flow coefficient
      Vx_U1,Vx_U2,Vx_U3 = phi*zeta_stator, phi, phi*zeta_rotor
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
      Ds_cp = -(1.0 - 1.0 / (1.0 - df.loc[index,'eta_lost'] )) * np.log(To_To1[-1])

      # Somewhat arbitrarily, split loss using loss ratio (default 0.5)
      s_cp = np.hstack((0.0, loss_ratio, 1.0)) * Ds_cp

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
      
      # Turbine
      Lam = (T_To1[2] - T_To1[1]) / (T_To1[2] - T_To1[0])
      
      if iterating==False:
         df.loc[index,'Al1'],df.loc[index,'Al2'],df.loc[index,'Al3'] = Al  #3
         df.loc[index,'Alrel1'],df.loc[index,'Alrel2'],df.loc[index,'Alrel3'] = Alrel  #3
         df.loc[index,'M1'],df.loc[index,'M2'],df.loc[index,'M3'] = Ma  #3
         df.loc[index,'M1rel'],df.loc[index,'M2rel'],df.loc[index,'M3rel'] = Marel  #3
         df.loc[index,'Ax1_Ax1'],df.loc[index,'Ax2_Ax1'],df.loc[index,'Ax3_Ax1'] = Dr_Drin  #3
         df.loc[index,'Po1_Po1'],df.loc[index,'Po2_Po1'],df.loc[index,'Po3_Po1'] = Po_Po1  #3
         df.loc[index,'To1_To1'],df.loc[index,'To2_To1'],df.loc[index,'To3_To1'] = To_To1  #3
         df.loc[index,'Vt1_U'],df.loc[index,'Vt2_U'],df.loc[index,'Vt3_U'] = Vt_U  #3
         df.loc[index,'Vt1rel_U'],df.loc[index,'Vt2rel_U'],df.loc[index,'Vt3rel_U'] = Vtrel_U  #3
         df.loc[index,'V1_U'],df.loc[index,'V2_U'],df.loc[index,'V3_U'] = V_U  #3
         df.loc[index,'V1rel_U'],df.loc[index,'V2rel_U'],df.loc[index,'V3rel_U'] = Vrel_U  #3
         df.loc[index,'P1_Po1'],df.loc[index,'P2_Po1'],df.loc[index,'P3_Po1'] = P_Po1  #3
         df.loc[index,'Po1rel_Po1'],df.loc[index,'Po2rel_Po1'],df.loc[index,'Po3rel_Po1'] = Porel_Po1  #3
         df.loc[index,'T1_To1'],df.loc[index,'T2_To1'],df.loc[index,'T3_To1'] = T_To1  #3
         df.loc[index,'mdot1_mdot1'],df.loc[index,'mdot2_mdot1'],df.loc[index,'mdot3_mdot1'] = mdot_mdot1  #3
         return df
      else:
         return Lam
      
   # Iteration step: returns error in reaction as function of exit yaw angle
   def iter_Al(x):
      Lam_guess = vars_from_Al(x,index,dataframe,iterating=True)

      return Lam_guess - dataframe.loc[index,'Lambda'] 
   
   for index, row in dataframe.iterrows():

      if iterate==True:
         # Solving for Lam in general is tricky
         # Our strategy is to map out a coarse curve first, pick a point
         # close to the desired reaction, then Newton iterate

         # Evaluate guesses over entire possible yaw angle range
         Al_guess = np.linspace(-89.0, 89.0, 21)
         Lam_guess = np.zeros_like(Al_guess)

         # Catch errors if this guess of angle is horrible/non-physical
         for i in range(len(Al_guess)):
            with np.errstate(invalid="ignore"):
               try:
                     Lam_guess[i] = iter_Al(Al_guess[i])
               except (ValueError, FloatingPointError):
                     Lam_guess[i] = np.nan

         # Remove invalid values
         Al_guess = Al_guess[~np.isnan(Lam_guess)]
         Lam_guess = Lam_guess[~np.isnan(Lam_guess)]

         # Trim to the region between minimum and maximum reaction
         # Now the slope will be monotonic
         i1, i2 = np.argmax(Lam_guess), np.argmin(Lam_guess)
         Al_guess, Lam_guess = Al_guess[i1:i2], Lam_guess[i1:i2]

         # Start the Newton iteration at minimum error point
         i0 = np.argmin(np.abs(Lam_guess))
         Al_soln = sciop.newton(iter_Al, 
                              x0=Al_guess[i0], 
                              x1=Al_guess[i0 - 1]
                              )
         
         vars_from_Al(Al_soln,index,dataframe)
      
      else:
         dataframe = vars_from_Al(None,index,dataframe)

   return dataframe
