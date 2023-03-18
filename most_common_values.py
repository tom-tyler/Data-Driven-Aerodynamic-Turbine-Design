from turbine_design.data_tools import read_in_data
import matplotlib.pyplot as plt
import pandas as pd

factor = 5
data = read_in_data('5D',state_retention_statistics=True)

fig, [ax1,ax2] = plt.subplots(2,1)

data.hist(column=['psi'],
          grid=True,
          bins=50,
          ax=ax1)
data.hist(column=['phi'],
          grid=True,
          bins=50,
          ax=ax2)

psi_ref = 1.78
phi_ref = 0.81

ax1.set_ylabel('Frequency')
ax2.set_ylabel('Frequency')

ax1.axvline(psi_ref*(1+factor/100),
            color='r',
            linestyle='--',
            label=f'{factor}% bounds for $\\psi$={psi_ref:.2f}')
ax1.axvline(psi_ref*(1-factor/100),
            color='r',
            linestyle='--')
ax2.axvline(phi_ref*(1+factor/100),
            color='r',
            linestyle='--',
            label=f'{factor}% bounds for $\\phi$={phi_ref:.2f}')
ax2.axvline(phi_ref*(1-factor/100),
            color='r',
            linestyle='--')

ax1.legend()
ax2.legend()

plt.show()
