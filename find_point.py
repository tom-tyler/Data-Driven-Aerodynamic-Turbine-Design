import turbine_design.data_tools as tools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

factor=10
df = tools.read_in_large_dataset('5D',state_retention_statistics=True,factor=factor)
lower_factor = 1 - factor/100
upper_factor = 1 + factor/100
val=0.8
df = df[df["phi"] < upper_factor*val]
df = df[df["phi"] > lower_factor*val]
val=2.2
df = df[df["psi"] < upper_factor*val]
df = df[df["psi"] > lower_factor*val]
val=0.5
df = df[df["Lambda"] < upper_factor*val]
df = df[df["Lambda"] > lower_factor*val]
val=0.5
df = df[df["M2"] < upper_factor*val]
df = df[df["M2"] > lower_factor*val]
val=0.65
df = df[df["Co"] < upper_factor*val]
df = df[df["Co"] > lower_factor*val]
print(df[['phi','psi','Lambda','Co','M2','runid']])

print(df['Yp_rotor']/(df['Yp_rotor']+df['Yp_stator']))
print(df['Yp_rotor'],df['Yp_stator'])
runid=float(df["runid"])
print(f'{runid:.20f}')