def read_in_data(path='Data',
                 dataset='all',
                 column_names=None
                 ):
   
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
      if column_names==None:
         if len(df.columns)==7:
            df.columns=["phi", "psi", "Lambda", "M", "Co", "eta_lost","runid"]
            dataframe_dict[data_name] = df
            
         elif len(df.columns)==6: #back-compatibility
            df.columns=["phi", "psi", "Lambda", "M", "Co", "eta_lost"]
            df["runid"] = 0
            dataframe_dict[data_name] = df
            
            
         else:
            print('error, invalid csv')
            quit()
      else:
         df.columns=column_names
         dataframe_dict[data_name] = df

   dataframe_list = dataframe_dict.values()   
   data = pd.concat(dataframe_list,ignore_index=True)
   return data