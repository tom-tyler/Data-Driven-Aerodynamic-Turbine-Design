from extra_params import turbine_params
import matplotlib.pyplot as plt
from dd_turb_design import read_in_large_dataset, split_data
import numpy as np
from sklearn.metrics import r2_score 

data = read_in_large_dataset(state_retention_statistics=True)

rand_data,other=split_data(data,0.3,random_seed_state=9)

turb = turbine_params(rand_data['phi'],rand_data['psi'],rand_data['M2'],rand_data['Co'])

turb.get_eta_lost()
x = turb.eta_lost
y = rand_data['eta_lost']
plt.scatter(x,y,marker='x')
plt.plot(x,x,color='r')
plt.show()



R_square = r2_score(x, y) 
print('Coefficient of Determination', R_square) 