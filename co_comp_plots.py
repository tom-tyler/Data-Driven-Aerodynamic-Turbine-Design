from turbine_design import turbine_GPR, read_in_data
import pandas as pd

data = read_in_data('5D only')
# data2 = read_in_data('2D only')

# data = pd.concat([data1, data2])

model = turbine_GPR()

model.fit(data,['phi','psi','Lambda','Co','M2'],
          'eta_lost')

model.optimised_kernel

model.plot('Co',constants={'phi':0.81,
                           'psi':1.78,
                           'M2':0.7,
                           'Lambda':0.5})