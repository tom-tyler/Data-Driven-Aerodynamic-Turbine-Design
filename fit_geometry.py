from dd_turb_design import turbine_GPR, read_in_data,read_in_large_dataset, split_data
import numpy as np
from itertools import combinations
import pandas as pd
from extra_params import turbine_info_4D
import matplotlib.pyplot as plt

data = read_in_large_dataset(state_retention_statistics=True)

traindf,testdf=split_data(data)

# training_inputs = ["phi",
#                     "psi", 
#                     "M2", 
#                     "Co", 
#                     's_cx_stator',
#                     's_cx_rotor',
#                     'Al2a',
#                     'Al3',
#                     'stagger_stator',
#                     'stagger_rotor',
#                     'Yp_stator',
#                     'Yp_rotor']

training_inputs = ["phi",
                    "psi", 
                    "M2", 
                    "Co", 
                    'Al2a',
                    'Al3',
                    'Yp_stator',
                    'Yp_rotor']

training_inputs_stator = ["phi",
                          "psi", 
                          "M2", 
                          "Co",
                          's_cx_stator',
                          'stagger_stator',
                          'Al2a',
                          'Al3']

training_inputs_rotor = ["phi",
                         "psi", 
                         "M2", 
                         "Co",
                         's_cx_rotor',
                         'stagger_rotor',
                         'Al2a',
                         'Al3']

inputs = ['M2','s_cx_stator','stagger_stator','Al2a']

turb_info = turbine_info_4D(traindf['phi'],
                            traindf['psi'],
                            traindf['M2'],
                            traindf['Co'])
traindf['stagger_stator'] = np.array(turb_info.get_stagger())[0]
traindf['s_cx_stator'] = np.array(turb_info.get_s_cx())[0]
traindf['Al2a'] = turb_info.get_Al()[1]

# inputs.extend(['Al3'])
output_variable = 'Yp_stator'
training_inputs = inputs

def pre_trained(var_list):
  var_list = list(var_list)

  if (inputs[0] in var_list) and (inputs[1] in var_list) and (inputs[2] in var_list) and (inputs[3] in var_list):
    return True
  else:
    return False
  
multidim_scoremax = 0
multidim_max_row = None
max_per_dim_list = []
dim_list = [4,5,6]
total_comb = 0
for dimensions in dim_list:
  comb = list(combinations(training_inputs,dimensions))
  num_comb = len(comb)
  total_comb += num_comb
print(f'total_comb = {total_comb}')

iterations = 0
for j,dimensions in enumerate(dim_list):
  comb = list(combinations(training_inputs,dimensions))
  #filter
  # comb=list(filter(pre_trained,comb))
  num_comb = len(comb)
  header_list = []
  for header in range(dimensions):
    header_list.append(f'var{header+1}')

  train_vars_df = pd.DataFrame(data=comb,
                              columns=header_list)

  scores = np.zeros(num_comb)
  train_vars_df['score'] = scores

  for i,training_vars in enumerate(comb):
    iterations+=1
    model = turbine_GPR()
    model.fit(traindf,
              variables=training_vars,
              output_key=output_variable,
              number_of_restarts=0,           
              length_bounds=[1e-3,1e4],
              noise_magnitude=1e-5,
              noise_bounds='fixed',
              nu='optimise', 
              overwrite=False)

    model.predict(testdf,
                  include_output=True)
    
    score = model.score
    scores[i] = score
    scoremax = np.max(np.array(scores))
    
    train_vars_df['score'] = scores

    max_indices = np.where(scores == scoremax)
    max_row = train_vars_df.iloc[max_indices]
    current_row = train_vars_df.iloc[np.array([i])]
    if scoremax>multidim_scoremax:
      multidim_scoremax = scoremax
      multidim_max_row = max_row
    print('-----------------------')
    print(model.optimised_kernel)
    print('current row:\n',current_row)
    print(' ')
    print('max row:\n',multidim_max_row)
    print(' ')
    print(f'Iteration {100*iterations/total_comb:.4g} % complete')
  max_per_dim_list.append(multidim_max_row)
print('max row:\n',multidim_max_row)    
print(max_per_dim_list)
# print(scores)
# print(np.max(np.array(scores)))

model.plot_accuracy(testdf,
                    line_error_percent=10,
                    identify_outliers=False)
