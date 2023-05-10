import os
import pandas as pd
import sys


def read_in_data(dataset='4D',
                 path='Data',
                 factor=5,
                 state_retention_statistics=False,
                 ignore_incomplete=False
                 ):
   """_summary_

   Args:
       dataset (str, optional): _description_. Defaults to '4D'.
       path (str, optional): _description_. Defaults to 'Data'.
       factor (int, optional): _description_. Defaults to 5.
       state_retention_statistics (bool, optional): _description_. Defaults to False.
       ignore_incomplete (bool, optional): _description_. Defaults to False.

   Returns:
       _type_: _description_
   """
   
   dataframe_dict = {}
   n_before = 0
   n_after = 0
   for filename in os.listdir(path):
      data_name = str(filename)[:-4]
      
      if data_name == 'turbine_data':
         continue
      
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
      elif dataset in ['2D','3D','4D','5D','2D_tip_gap']:
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
         
      elif dataset in ['3D']:
         # Co = 0.65
         val=0.65
         df = df[df["Co"] < upper_factor*val]
         df = df[df["Co"] > lower_factor*val]
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
         
      elif dataset in ['2D_tip_gap']:
         # Lambda = 0.5
         val=0.5
         df = df[df["Lambda"] < upper_factor*val]
         df = df[df["Lambda"] > lower_factor*val]
         # M2 = 0.7 or 0.65
         val=0.67
         df = df[df["M2"] < upper_factor*val]
         df = df[df["M2"] > lower_factor*val]
         # Co = 0.65 or 0.7
         val=0.65
         df = df[df["Co"] < upper_factor*val]
         df = df[df["Co"] > lower_factor*val]
         
      df = df.reindex(sorted(df.columns), axis=1)
         
      n_after += len(df.index)
      
      dataframe_dict[data_name] = df

   if state_retention_statistics==True:
      print(f'n_before = {n_before}\nn_after = {n_after}\n%retained = {n_after/n_before*100:.2f} %')
   dataframe_list = dataframe_dict.values()   
   data = pd.concat(dataframe_list,ignore_index=True)

   return data

def read_in_large_dataset(dataset='4D',
                          data_filename='Data/turbine_data.csv',
                          factor=5,
                          state_retention_statistics=False
                          ):
   """_summary_

   Args:
       dataset (str, optional): _description_. Defaults to '4D'.
       data_filename (str, optional): _description_. Defaults to 'Data/turbine_data.csv'.
       factor (int, optional): _description_. Defaults to 5.
       state_retention_statistics (bool, optional): _description_. Defaults to False.

   Returns:
       _type_: _description_
   """

   df = pd.read_csv(data_filename)

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
               'Al3',
               'tau_c',
               'fc_1',
               'fc_2',
               'htr',
               'spf_stator',
               'stagger_stator',
               'recamber_le_stator',
               'recamber_te_stator',
               'Rle_stator',
               'beta_stator',
               't_ps_stator',
               't_ss_stator',
               'max_t_loc_ps_stator',
               'max_t_loc_ss_stator',
               'lean_stator',
               'spf_rotor',
               'stagger_rotor',
               'recamber_le_rotor',
               'recamber_te_rotor',
               'Rle_rotor',
               'beta_rotor',
               't_ps_rotor',
               't_ss_rotor',
               'max_t_loc_ps_rotor',
               'max_t_loc_ss_rotor',
               'lean_rotor']
      
   # filter by factor% error
   lower_factor = 1 - factor/100
   upper_factor = 1 + factor/100
   
   n_before = len(df.index)
   
   df = df[df['tau_c'] == 0.0]
   df = df[df['fc_1'] == 0.0]
   df = df[df['fc_2'] == 0.0]

   if dataset in ['4D']:
      # Lambda = 0.5
      val = 0.5
      df = df[df["Lambda"] < upper_factor*val]
      df = df[df["Lambda"] > lower_factor*val]
      
   elif dataset in ['3D']:
      # Co = 0.65
      val=0.65
      df = df[df["Co"] < upper_factor*val]
      df = df[df["Co"] > lower_factor*val]
      # Lambda = 0.5
      val = 0.5
      df = df[df["Lambda"] < upper_factor*val]
      df = df[df["Lambda"] > lower_factor*val]
      
   elif dataset in ['2D']:
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
      
   elif dataset in ['2D_tip_gap']:
      # Lambda = 0.5
      val=0.5
      df = df[df["Lambda"] < upper_factor*val]
      df = df[df["Lambda"] > lower_factor*val]
      # M2 = 0.7 or 0.65
      val=0.67
      df = df[df["M2"] < upper_factor*val]
      df = df[df["M2"] > lower_factor*val]
      # Co = 0.65 or 0.7
      val=0.65
      df = df[df["Co"] < upper_factor*val]
      df = df[df["Co"] > lower_factor*val]
      
   df = df.reindex(sorted(df.columns), axis=1)
      
   n_after = len(df.index)

   if state_retention_statistics==True:
      print(f'n_before = {n_before}\nn_after = {n_after}\n%retained = {n_after/n_before*100:.2f} %')

   return df

def split_data(df,
               fraction_training=0.75,
               random_seed_state=2
               ):
   """_summary_

   Args:
       df (_type_): _description_
       fraction_training (float, optional): _description_. Defaults to 0.75.
       random_seed_state (int, optional): _description_. Defaults to 2.

   Returns:
       _type_: _description_
   """
   
   training_data = df.sample(frac=fraction_training,
                             random_state=random_seed_state
                             )
   testing_data = df.loc[~df.index.isin(training_data.index)]
   return training_data,testing_data
