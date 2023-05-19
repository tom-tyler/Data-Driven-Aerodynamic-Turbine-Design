from turbine_design.data_tools import read_in_data
import matplotlib.pyplot as plt
import pandas as pd

factor = 5
data = read_in_data('4D only',state_retention_statistics=True)

fig, [ax1,ax2,ax3,ax4] = plt.subplots(1,4,sharey=True)
bins=25

data.hist(column=['phi'],
          grid=True,
          bins=bins,
          ax=ax1)
data.hist(column=['psi'],
          grid=True,
          bins=bins,
          ax=ax2)
data.hist(column=['M2'],
          grid=True,
          bins=bins,
          ax=ax3)
data.hist(column=['Co'],
          grid=True,
          bins=bins,
          ax=ax4)

psi_ref = 1.78
phi_ref = 0.81

ax1.set_ylabel('Frequency')

# ax1.axvline(psi_ref*(1+factor/100),
#             color='r',
#             linestyle='--',
#             label=f'{factor}% bounds for $\\psi$={psi_ref:.2f}')
# ax1.axvline(psi_ref*(1-factor/100),
#             color='r',
#             linestyle='--')
# ax2.axvline(phi_ref*(1+factor/100),
#             color='r',
#             linestyle='--',
#             label=f'{factor}% bounds for $\\phi$={phi_ref:.2f}')
# ax2.axvline(phi_ref*(1-factor/100),
#             color='r',
#             linestyle='--')

# ax1.legend()
# ax2.legend()
fig.tight_layout()
plt.show()
