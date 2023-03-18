from turbine_design.data_tools import read_in_large_dataset
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import numpy as np

data = read_in_large_dataset(state_retention_statistics=True)

variables = ['phi',
             'psi',
             'M2',
             'Co']
variable_vals = [[0.6,0.7,0.8,0.95],
                 [1.2,1.5,1.8,2.1],
                 [0.5,0.6,0.7,0.8],
                 [0.55,0.6,0.65,0.7]]
bin_size = 15

params = ["eta_lost",
            'Yp_stator', 
            'Yp_rotor', 
            'zeta_stator',
            'zeta_rotor',
            's_cx_stator',
            's_cx_rotor',
            'loss_rat',
            'Al1',
            'Al2a',
            'Al2b',
            'Al3',
            'stagger_stator',
            'recamber_te_stator',
            'Rle_stator',
            'beta_stator',
            't_ps_stator',
            't_ss_stator',
            'max_t_loc_ps_stator',
            'max_t_loc_ss_stator',
            'lean_stator',
            'stagger_rotor',
            'recamber_te_rotor',
            'Rle_rotor',
            'beta_rotor',
            't_ps_rotor',
            't_ss_rotor',
            'max_t_loc_ps_rotor',
            'max_t_loc_ss_rotor']
params=['Rle_stator']
for param in params:

    fig, axes = plt.subplots(len(variable_vals[0]),len(variables),sharex=True)


    for indices, axis in np.ndenumerate(axes):
        (i,j) = indices
        data_i = data[(data[variables[j]]>variable_vals[j][i]*0.95) & (data[variables[j]]<variable_vals[j][i]*1.05)]

        data_i.hist(column=[param],
                    grid=True,
                    bins=bin_size,
                    ax=axis,
                    label=f'{variables[j]}={variable_vals[j][i]}')
        
        axis.set_ylabel(f'Frequency {variables[j]}')
        axis.legend(fontsize=7)

    fig.set_figheight(8)
    fig.set_figwidth(12)
    fig.tight_layout()
    plt.show()

    # fig.savefig(f'output_histograms/{param}_histogram_4x4.svg',
    #             format='svg')

    print('saved ',param)

