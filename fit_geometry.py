from dd_turb_design import turbine_GPR, read_in_data,read_in_large_dataset, split_data
import numpy as np

data = read_in_large_dataset(state_retention_statistics=True)

traindf,testdf=split_data(data)

model = turbine_GPR() #0.437 #0.453
model.fit(traindf,
          variables=['psi',
                    'Co',
                    'M2',
                    'phi'],
          output_key='t_ss_stator',
           number_of_restarts=0,           
           length_bounds=[1e-3,1e7],
           noise_magnitude=1e-3,
           noise_bounds=[1e-9,1e-1],
          nu='optimise',
          overwrite=False)

print(model.optimised_kernel)

model.plot_accuracy(testdf,
                    line_error_percent=10,
                    identify_outliers=False)