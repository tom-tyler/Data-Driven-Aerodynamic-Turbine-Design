# Call this script using
# $ python post_process_example.py HDF5_FILE
"""Post process a steady Turbostream solution"""

import sys, os
import turbigen.turbostream.grid as grid
import matplotlib.pyplot as plt

import numpy as np

# Load the grid file provided on command line
# output_hdf5 = sys.argv[1]
# g = grid.read_hdf5(output_hdf5)

# Load grid file manually by typing in runid
job_name = str(input('Enter name of job results (e.g. data-A results) : '))
datapoint_runid = str(input('Enter run id (e.g. 002803602529): '))

datapoint_directory = os.path.join(job_name, datapoint_runid)
datapoint_hdf5 = os.path.join(os.path.join(datapoint_directory, datapoint_runid), 'output_avg.hdf5')

if os.path.isfile(datapoint_hdf5):
    g = grid.read_hdf5(datapoint_hdf5)
else:
    print('incorrect input')
    quit()


# Extract cut (r, rt) or (y, z) planes upstream and downstream of each row
(stator_in, stator_out), (rotor_in, rotor_out) = g.cut_rows()

# Calculate flow coefficient
phi = rotor_in.mix_out().vx / rotor_in.mix_out().U
print("Flow coefficient, phi = %.3f" % phi)

# Make a contour plot of flow coefficient distribution
fig, ax = plt.subplots()
hc = ax.contourf(
    stator_out.y,
    stator_out.z,
    stator_out.vx / rotor_in.mix_out().U,
    np.linspace(0.8, 1.2, 11),
)
ax.axis("equal")
plt.colorbar(hc)
plt.savefig(os.path.join(datapoint_directory,"contour_phi.pdf"))

# Now extract cut around the blade surfaces
vane, blade = g.cut_blade_surfs()

# Plot rotor geometry at hub, mid, tip
fig, ax = plt.subplots()
for jplot in (0, 33, -1):
    ax.plot(blade.x[:, jplot], blade.rt[:, jplot])
ax.axis("equal")
plt.savefig(os.path.join(datapoint_directory,"geometry.pdf"))

# Cut at mid-height (x, rt) plane
C = g.cut_span(0.5)

# Plot contours of density
fig, ax = plt.subplots()
lev_rho = np.linspace(2.2, 3.5, 11)
for c in C:
    hc = ax.contourf(c.x, c.rt, c.ro, lev_rho)
ax.axis("equal")
plt.colorbar(hc)
plt.savefig(os.path.join(datapoint_directory,"contour_rho.pdf"))