# Call this script using
# $ python post_process_example.py
"""Post process a steady Turbostream solution"""

import sys, os
import turbigen.turbostream.grid as grid
import matplotlib.pyplot as plt

import numpy as np

def variable_to_parameter(variable, section):
    if variable == 'M':
        return section.M
    elif variable == 'ro'
        return section.ro
    

def plot_contour_XT(variable,cut_height=0.5,num_levels = 11):
    
    # Cut in (x, rt) plane
    plane = g.cut_span(cut_height)
    
    # Plot contours
    fig, ax = plt.subplots()
    
    # Get max and min values for contours
    min_val,max_val=[],[]
    for section in plane:
        min_val.append(np.amin(section.ro))
        max_val.append(np.amax(section.ro))
        
    min_val=np.amin(min_val)
    max_val=np.amax(max_val)

    levels = np.linspace(min_val,max_val,num_levels)
    
    for section in plane:
        hc = ax.contourf(section.x, section.rt, section.ro, levels)
    ax.axis("equal")
    plt.colorbar(hc)
    plot_name = 'contour_'+'ro'+'.pdf'
    plt.savefig(os.path.join(plot_directory,plot_name))
    print('Plotted: ')

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

else:
    print('incorrect input')
    quit()
    
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)

num_levels=11

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
for c in C:
    lev_rho = np.linspace(np.amin(c.ro), np.amax(c.ro), num_levels)
    hc = ax.contourf(c.x, c.rt, c.ro, lev_rho)
ax.axis("equal")
plt.colorbar(hc)
plt.savefig(os.path.join(plot_directory,"contour_rho.pdf"))

# Plot contours of Mach number
plot_contour_XT()

print('Figures stored in: ', plot_directory)