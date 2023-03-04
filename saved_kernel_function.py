   def __init__(self):
      
      saved_kernel_values = pd.read_csv('kernel_parameters.csv')

      saved_dimensions=saved_kernel_values.iloc[0]
      saved_length=[]
      saved_lower_bound_length=[]
      saved_upper_bound_length=[]
      for i in range(len(saved_dimensions)):
         saved_length.append(saved_kernel_values.iloc[i+1])
         saved_lower_bound_length.append(saved_kernel_values.iloc[i+1+saved_dimensions])
         saved_upper_bound_length.append(saved_kernel_values.iloc[i+1+2*saved_dimensions])
      saved_lengthb = zip(saved_lower_bound_length,saved_upper_bound_length)
      saved_nu = saved_kernel_values.iloc[saved_dimensions*3+2]
      saved_noise = saved_kernel_values.iloc[saved_dimensions*3+3]
      saved_noiseb = [saved_kernel_values.iloc[saved_dimensions*3+4],saved_kernel_values.iloc[saved_dimensions*3+5]]
      
      noise_kernel = kernels.WhiteKernel(noise_level=saved_noise,
                                         noise_level_bounds=saved_noiseb)

      kernel_form = self.matern_kernel(int(saved_dimensions),bounds=saved_lengthb) + noise_kernel
      
      gaussian_process = GaussianProcessRegressor(kernel=kernel_form,
                                                  n_restarts_optimizer=0,
                                                  random_state=0
                                                  )
      
      gaussian_process.set_params(kernel__k1__length_scale=saved_length)
      gaussian_process.set_params(kernel__k1__length_scale_bounds=saved_lengthb)
      gaussian_process.set_params(kernel__k1__nu=saved_nu)
      gaussian_process.set_params(kernel__k2__noise_level=saved_noise)
      gaussian_process.set_params(kernel__k2__noise_level_bounds=saved_noiseb)
      
      with open("df_headers.txt") as file:
         headers = [str(line.strip()) for line in file.readlines()]
      self.output_key = headers[-1]
      self.variables = headers[:-1]
         
      training_dataframe = pd.read_csv('training_dataframe.csv',names=headers)
      self.input_array_train = training_dataframe.drop(columns=[self.output_key])
      self.output_array_train = training_dataframe[self.output_key]
      self.fit_dimensions = saved_dimensions
      self.no_points = len(training_dataframe.index)
      
      self.limit_dict = {}
      for column in self.input_array_train:
         self.limit_dict[column] = (np.around(training_dataframe[column].min(),decimals=1),
                                    np.around(training_dataframe[column].max(),decimals=1)
                                    )
         
      self.fitted_function = gaussian_process
      self.optimised_kernel = self.fitted_function.kernel_
         
      self.min_train_output = np.min([self.output_array_train])
      self.max_train_output = np.max([self.output_array_train])