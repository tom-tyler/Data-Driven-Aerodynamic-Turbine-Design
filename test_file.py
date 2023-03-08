from extra_params import turbine_info_4D
import numpy as np

print(np.array(5))

turb = turbine_info_4D(0.8,1.8,0.7,0.65)

print(turb.dim_from_omega(314,1600,160000))
print(turb.dim_from_omega(100,1600,100000))
print(turb.dim_from_omega(314,1600,160000))
print(turb.dim_from_mdot(314,1600,160000))
