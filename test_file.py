from extra_params import turbine_info_4D

turb = turbine_info_4D(0.8,1.8,0.7,0.65)

print(turb.dim_from_mdot(1,1600,150000))