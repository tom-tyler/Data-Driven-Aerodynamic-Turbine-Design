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
datapoint_runid = str(input('Enter run id (e.g. 002803602529): '))

for root, dirs, files in os.walk('Results'):

    if datapoint_runid in dirs:
        job_directory = root
        print('Datapoint is in: ',job_directory)
        datapoint_directory = os.path.join(job_directory, datapoint_runid)
        break

datapoint_hdf5 = os.path.join(datapoint_directory, os.path.join(datapoint_runid, 'output_avg.hdf5'))
print('HDF5 path: ', datapoint_hdf5)

if os.path.isfile(datapoint_hdf5):
    g = grid.read_hdf5(datapoint_hdf5)
    plot_directory = os.path.join(datapoint_directory, 'Figures')
    os.makedirs(plot_directory)
else:
    print('incorrect input')
    quit()

num_levels = 11

# Extract cut (r, rt) or (y, z) planes upstream and downstream of each row
(stator_in, stator_out), (rotor_in, rotor_out) = g.cut_rows()

# Calculate flow coefficient
phi = stator_out.vx / rotor_in.mix_out().U

# Make a contour plot of flow coefficient distribution
fig, ax = plt.subplots()
hc = ax.contourf(
    stator_out.y,
    stator_out.z,
    phi,
    np.linspace(np.amin(phi), np.amax(phi), num_levels),
)
ax.axis("equal")
plt.colorbar(hc)
plt.savefig(os.path.join(plot_directory,"contour_phi.pdf"))

# Now extract cut around the blade surfaces
vane, blade = g.cut_blade_surfs()

# Plot rotor geometry at hub, mid, tip
fig, ax = plt.subplots()
for jplot in (0, 33, -1):
    ax.plot(blade.x[:, jplot], blade.rt[:, jplot])
ax.axis("equal")
plt.savefig(os.path.join(plot_directory,"geometry.pdf"))

# Cut at mid-height (x, rt) plane
C = g.cut_span(0.5)

# Plot contours of density
fig, ax = plt.subplots()
lev_rho = np.linspace(np.amin(c.ro), np.amax(c.ro), num_levels)
for c in C:
    hc = ax.contourf(c.x, c.rt, c.ro, lev_rho)
ax.axis("equal")
plt.colorbar(hc)
plt.savefig(os.path.join(plot_directory,"contour_rho.pdf"))

print('Figures stored in: ', plot_directory)
