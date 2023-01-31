        
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
                 plotting_grid=False,
                 superpose_variable=None
                 ):
      
      if axis == None:
         fig,axis = plt.subplots(1,1,sharex=True,sharey=True)
         plot_now = True
      else:
         plot_now = False
         
      if superpose_variable!=None:
         superpose_parameters=True
         
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
                      CI_percent=CI_percent,
                      superpose_parameters=superpose_parameters)
         
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
         
         
         axis.plot(x1, 
                   self.mean_prediction, 
                   label=r"Mean prediction", 
                   color='blue'
                   )
         
         # if superpose_variable != None:
         #    axis.plot(self.predicted_dataframe[superpose_variable],
         #              self.mean_prediction, 
         #              label=superpose_variable, 
         #              color='green'
         #              )
         
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
         fig.suptitle(f'n = {self.no_points}')
         fig.tight_layout()
         plt.show()
      