from turbine_design import turbine_GPR, read_in_data, split_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = read_in_data('5D')

datatr,datate = split_data(data)

model = turbine_GPR('eta_lost_5D')

factor=0.11

# vall=0.4
# valu=0.6
# range = valu-vall
# datate = datate[datate["Lambda"] < valu-range*factor]
# datate = datate[datate["Lambda"] > vall+range*factor]

# vall=0.5
# valu=0.95
# range = valu-vall
# datate = datate[datate["M2"] < valu-range*factor]
# datate = datate[datate["M2"] > vall+range*factor]

vall=0.4
valu=0.8
range = valu-vall
datate = datate[datate["Co"] < valu-range*factor]
datate = datate[datate["Co"] > vall+range*factor]

# vall=0.4
# valu=1.2
# range = valu-vall
# datate = datate[datate["phi"] < valu-range*factor]
# datate = datate[datate["phi"] > vall+range*factor]

# vall=1.0
# valu=2.4
# range = valu-vall
# datate = datate[datate["psi"] < valu-range*factor]
# datate = datate[datate["psi"] > vall+range*factor]

model.plot_accuracy(datate,
                    line_error_percent=10,
                    legend_outside=True,
                    identify_outliers=True)

# buffer = np.arange(-10,20,5)

# r2 = [0.842,
#       0.896,
#       0.896,
#       0.893,
#       0.894,
#       0.888,
#       0.897,
#       0.887,
#       0.916,
#       0.867,
#       0.933,
#       0.927]

# r2 = [0.842,
#       0.896,
#       0.894,
#       0.897,
#       0.916,
#       0.933]

# plt.grid()
# plt.xlabel('Buffer to edge of sample space (%)')
# plt.ylabel('$R^2$')
# plt.scatter(buffer,r2,marker='x',color='darkviolet',s=70,zorder=5)
# plt.ylim(0.825,0.95)
# plt.show()