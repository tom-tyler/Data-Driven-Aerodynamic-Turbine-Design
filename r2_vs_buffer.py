from turbine_design import turbine_GPR, read_in_data, split_data, read_in_large_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = read_in_data('4D')

datatr,datate = split_data(data)

model = turbine_GPR('eta_lost_4D')

# model.fit(datatr,
#           ['phi','psi','M2','Co'],
#           'eta_lost',
#           model_name='eta_lost_only4d')

# fig,ax = model.plot('phi','psi',
#            contour_step=0.01,
#            constants={'Co':0.66,'M2':0.67},
#            fix_eta_lost_colors=True,
#            num_points=500,
#            limit_dict={'phi':(0.4,1.2),
#                        'psi':(1.0,2.4),
#                        'M2':(0.5,0.95),
#                        'Co':(0.4,0.8)},
#            show=False)

# fig.set_figwidth(4)
# fig.set_figheight(4)
# fig.tight_layout()

# plt.show()
factor=0

vall=0.475
valu=0.525
range = valu-vall
datate = datate[datate["Lambda"] < valu-range*factor]
datate = datate[datate["Lambda"] > vall+range*factor]

vall=0.5
valu=0.95
range = valu-vall
datate = datate[datate["M2"] < valu-range*factor]
datate = datate[datate["M2"] > vall+range*factor]

vall=0.5
valu=0.8
range = valu-vall
datate = datate[datate["Co"] < valu-range*factor]
datate = datate[datate["Co"] > vall+range*factor]

vall=0.4
valu=1.2
range = valu-vall
datate = datate[datate["phi"] < valu-range*factor]
datate = datate[datate["phi"] > vall+range*factor]

vall=1.0
valu=2.4
range = valu-vall
datate = datate[datate["psi"] < valu-range*factor]
datate = datate[datate["psi"] > vall+range*factor]

model.plot_accuracy(datate,
                    line_error_percent=10,
                    equal_axis=True)

# # buffer = np.arange(-10,20,5)
# buffer = np.arange(0,20,2.5)

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
#       0.888,
#       0.897,
#       0.887,
#       0.916,
#       0.889,
#       0.933,
#       0.927]

# # r2 = [0.842,
# #       0.896,
# #       0.894,
# #       0.897,
# #       0.916,
# #       0.933]

# buffer = [0.40,
#           0.425,
#           0.45,
#           0.475,
#           0.50,
#           0.525,
#           0.55,
#           0.575,
#           0.60]

# r2 = [0.842,
#       0.889,
#       0.924,
#       0.952,
#       0.952,
#       0.953,
#       0.954,
#       0.964,
#       0.967]

# plt.grid()
# plt.xlabel('Lower $C_0$ design space limit')
# plt.ylabel('$R^2$')
# plt.scatter(buffer,r2,marker='x',color='darkviolet',s=70,zorder=5)
# # plt.ylim(0.825,0.96)
# plt.show()