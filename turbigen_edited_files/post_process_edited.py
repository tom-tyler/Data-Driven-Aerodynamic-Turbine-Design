# $ python post_process_example.py
"""Post process a steady Turbostream solution"""

import sys, os
import turbigen.turbostream.grid as grid
import matplotlib.pyplot as plt

import numpy as np

def param(variable, section):
    if variable == 'M':
        return section.mach
    elif variable == 'Mrel':
        return section.mach_rel
    elif variable == 'V':
        return np.sqrt(section.vsq)
    elif variable == 'Vrel':
        return np.sqrt(section.vsq_rel)
    elif variable == 'ro':
        return section.ro
    elif variable == 'p':
        return section.pstat
    elif variable == 'T':
        return section.tstat
    elif variable == 'p0':
        return section.pstag
    elif variable == 'T0':
        return section.tstag
    elif variable == 'p0rel':
        return section.pstag_rel
    elif variable == 'T0rel':
        return section.tstag_rel
    elif variable == 'rovx':
        return section.rovx
    else:
        print(str(variable)+' not currently implimented')
    

def plot_contour_XT(variable,cut_height=0.5,num_levels = 11):
    
    # Cut in (x, rt) plane
    plane = g.cut_span(cut_height)
    
    # Plot contours
    fig, ax = plt.subplots()
    
    # Get max and min values for contours
    min_val,max_val=[],[]
    min_x,max_x=[],[]
    min_rt,max_rt=[],[]
    
    for section in plane:
        var = param(variable,section)
        min_val.append(np.amin(var))
        max_val.append(np.amax(var))
        min_x.append(np.amin(section.x))
        max_x.append(np.amax(section.x))
        min_rt.append(np.amin(section.rt))
        max_rt.append(np.amax(section.rt))
        
    min_val=np.amin(min_val)
    max_val=np.amax(max_val)
    min_x=np.amin(min_x)
    max_x=np.amax(max_x)
    min_rt=np.amin(min_rt)
    max_rt=np.amax(max_rt)

    levels = np.linspace(min_val,max_val,num_levels)
    
    for section in plane:
        var = param(variable,section)
        hc = ax.contourf(section.x, section.rt, var, levels)
    ax.axis("scaled")
    ax.set_title(variable+' plot at '+str(cut_height*100)+'% cut height')
    ax.set_xlim(min_x,max_x)
    ax.set_ylim(min_rt,max_rt)
    plt.colorbar(hc,orientation='horizontal',format='%.3g')
    plot_name = 'contour_'+variable+'.pdf'
    plt.savefig(os.path.join(plot_directory,plot_name))
    print('Plotted: ', variable)


    
def loss_breakdown():
    
    # Extract cut (r, rt) or (y, z) planes upstream and downstream of each row
    (stator_in, stator_out), (rotor_in, rotor_out) = g.cut_rows()
    

    #get rovx for stator_out,rotor_in and rotor_out and plot these like
    # how phi was plotted
    
    rovx = np.array(stator_out.rovx)
    RT = np.array(stator_out.rt)
    R = np.array(stator_out.r)
    rt = RT[0,:]
    r = R[:,0]
    
    rt_integral = np.trapz(rovx,rt,axis=1)
    total_mdot = np.trapz(rt_integral,r,axis=0)
    third_mdot = total_mdot/3
    mdot = 0
    r_low = 0
    r_high = 0
    r_hub = r[0]
    r_tip = r[-1]
    for i in range(len(r)-1):
        mdot += 0.5*(r[i+1]-r[i])*(rt_integral[i]+rt_integral[i+1])
        if (mdot > third_mdot) and (r_low==0):
            r_low = r[i]
        if (mdot > 2*third_mdot) and (r_high==0):
            r_high = r[i]
    print('r: ',r_hub,r_low,r_high,r_tip)
            

    fig, ax = plt.subplots()
    hc = ax.contourf(
        RT,
        R,
        rovx,
        np.linspace(np.amin(rovx), np.amax(rovx), num_levels))
    ax.axis("equal")
    ax.set_xlim(np.amin(rt),np.amax(rt))
    ax.set_ylim(np.amin(r),np.amax(r))
    plt.colorbar(hc)
    plot_name = "rovx_stator_out.pdf"
    plt.savefig(os.path.join(plot_directory,plot_name))
    print('Plotted: ', plot_name[:-4])
    
    rovx = np.array(rotor_in.rovx)
    RT = np.array(rotor_in.rt)
    R = np.array(rotor_in.r)
    rt = RT[0,:]
    r = R[:,0]
    
    rt_integral = np.trapz(rovx,rt,axis=1)
    total_mdot = np.trapz(rt_integral,r,axis=0)
    third_mdot = total_mdot/3
    mdot = 0
    r_low = 0
    r_high = 0
    r_hub = r[0]
    r_tip = r[-1]
    for i in range(len(r)-1):
        mdot += 0.5*(r[i+1]-r[i])*(rt_integral[i]+rt_integral[i+1])
        if (mdot > third_mdot) and (r_low==0):
            r_low = r[i]
        if (mdot > 2*third_mdot) and (r_high==0):
            r_high = r[i]
    print('r: ',r_hub,r_low,r_high,r_tip)
            

    fig, ax = plt.subplots()
    hc = ax.contourf(
        RT,
        R,
        rovx,
        np.linspace(np.amin(rovx), np.amax(rovx), num_levels))
    ax.axis("equal")
    ax.set_xlim(np.amin(rt),np.amax(rt))
    ax.set_ylim(np.amin(r),np.amax(r))
    plt.colorbar(hc)
    plot_name = "rovx_rotor_in.pdf"
    plt.savefig(os.path.join(plot_directory,plot_name))
    print('Plotted: ', plot_name[:-4])
    
    rovx = np.array(rotor_out.rovx)
    RT = np.array(rotor_out.rt)
    R = np.array(rotor_out.r)
    rt = RT[0,:]
    r = R[:,0]
    
    rt_integral = np.trapz(rovx,rt,axis=1)
    total_mdot = np.trapz(rt_integral,r,axis=0)
    third_mdot = total_mdot/3
    mdot = 0
    r_low = 0
    r_high = 0
    r_hub = r[0]
    r_tip = r[-1]
    for i in range(len(r)-1):
        mdot += 0.5*(r[i+1]-r[i])*(rt_integral[i]+rt_integral[i+1])
        if (mdot > third_mdot) and (r_low==0):
            r_low = r[i]
        if (mdot > 2*third_mdot) and (r_high==0):
            r_high = r[i]
    print('r: ',r_hub,r_low,r_high,r_tip)
            

    fig, ax = plt.subplots()
    hc = ax.contourf(
        RT,
        R,
        rovx,
        np.linspace(np.amin(rovx), np.amax(rovx), num_levels))
    ax.axis("equal")
    ax.set_xlim(np.amin(rt),np.amax(rt))
    ax.set_ylim(np.amin(r),np.amax(r))
    plt.colorbar(hc)
    plot_name = "rovx_rotor_out.pdf"
    plt.savefig(os.path.join(plot_directory,plot_name))
    print('Plotted: ', plot_name[:-4])


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
    stator_out.rt,
    stator_out.r,
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


# Plot contours of Mach number
plot_contour_XT('M')
plot_contour_XT('Mrel')

# Plot contours of velocity
plot_contour_XT('V')
plot_contour_XT('Vrel')

# Plot contours of density
plot_contour_XT('ro')

# Plot contours of pressure
plot_contour_XT('p')
plot_contour_XT('p0')
plot_contour_XT('p0rel')

# Plot contours of temperature
plot_contour_XT('T')
plot_contour_XT('T0')
plot_contour_XT('T0rel')

loss_breakdown()

print('Figures stored in: ', plot_directory)