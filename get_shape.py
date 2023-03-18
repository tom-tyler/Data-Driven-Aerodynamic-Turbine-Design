"""Given a params object, get geometry and plot at midspan."""
from turbine_design.turbigen import three_dimensional_stage, ohmesh
import numpy as np
import matplotlib.pyplot as plt
from stl import mesh
import sys

def get_shape(params):
    """Function to return shape from a parameters object.

    Returns
    -------
    nb: (nrow,) array
        Numbers of blades.
    h: (npts, 2) array
        x-r hub coordinates.
    c: (npts, 2) array
        x-r casing coordinates.
    ps: nested list ps[row][section][point on section, x/r/t]
        Pressure-side aerofoil coordinates.
    ss:
        Same as `ps` for the suction side.

    """

    # The meshing routines need "Section generators", callables that take a
    # span fraction and output a blade section
    def vane_section(spf):
        return params.interpolate_section(0, spf)

    def blade_section(spf):
        return params.interpolate_section(1, spf, is_rotor=True)

    sect_generators = [vane_section, blade_section]

    # For brevity
    Dstg = params.dimensional_stage

    # Evaluate shape
    _, _, nb, ps, ss, h, c, _ = ohmesh.get_sections_and_annulus_lines(
        params.dx_c, Dstg.rm, Dstg.Dr, Dstg.cx, Dstg.s, params.tau_c, sect_generators
        )

    return nb, h, c, ps, ss


# Load a parameter set
params = three_dimensional_stage.StageParameterSet.from_json("turbine_design/turbine_json/turbine_params.json")

# Extract shape
nb, h, c, ps, ss = get_shape(params)

blade = 'stator'

if blade=='stator':
    irow = 0
elif blade=='rotor':
    irow = 1
else:
    sys.exit('choose stator or rotor blade')

vertices = np.array([]).reshape(0,3)
faces = np.array([],dtype=np.int64).reshape(0,3)
ri_len = len(ps[irow])
for ri in range(ri_len):
        
    sect_ps = ps[irow][ri]
    x_ps, r_ps, t_ps = sect_ps.T
    rt_ps = r_ps*t_ps
    sect_ss = ss[irow][ri]
    x_ss, r_ss, t_ss = sect_ss.T
    rt_ss = r_ss*t_ss
    x = np.concatenate((x_ps, np.flip(x_ss)), axis=None)
    rt = np.concatenate((rt_ps, np.flip(rt_ss)), axis=None)
    r = np.concatenate((r_ps, np.flip(r_ss)), axis=None)
    vertices_i = np.column_stack((x,rt,r))
    vertices = np.vstack([vertices,vertices_i])
    n = len(x)
    
    if ri==0:
        # THIS SECTION IS CORRECT!! DO NOT CHANGE
        faces_i = [0]*(n-1)
        for i in range(n-1):    
            faces_i[i] = [int(i+1),
                          int(n-1-i),
                          int(i)]

        faces_i = np.array(faces_i,dtype=np.int64)
        faces = np.vstack([faces,faces_i])
        
    elif ri==ri_len-1:
        # THIS SECTION IS CORRECT!! DO NOT CHANGE
        faces_i = [0]*(n-1)
        for i in range(n*ri,n*(ri+1)-1):    
            faces_i[i-n*ri] = [int(i+1),
                               int(n*ri-1-i),
                               int(i)]

        faces_i = np.array(faces_i,dtype=np.int64)
        faces = np.vstack([faces,faces_i])
        
    else:
        # THIS SECTION IS CORRECT!! DO NOT CHANGE
        faces_i = [0]*(n-1)*2
        for i in range(n*ri,n*(ri+1)-1):    
            faces_i[i-n*ri] = [int(i),
                               int(i+1-n),
                               int(i+1)]
            faces_i[i-n*(ri-1)-1] = [int(i),
                                     int(i-n),
                                     int(i+1-n)]    

        faces_i = np.array(faces_i,dtype=np.int64)
        faces = np.vstack([faces,faces_i])

# Create the mesh
turbine_blade = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        turbine_blade.vectors[i][j] = vertices[f[j],:]

# Write the mesh to file "cube.stl"
turbine_blade.save(f'turbine_{blade}_blade.stl')

# Plot
jmid = 40  # At midspan (approx)
fig, ax = plt.subplots()
for irow in range(2):
    tex = [0,0]
    tert = [0,0]
    for i,side in enumerate([ps, ss]):
        sectmid = side[irow][jmid]
        x, r, t = sectmid.T
        rt = r*t
        ax.plot(x, rt, '-b')
        tex[i] = x[-1]
        tert[i] = rt[-1]
    ax.plot(tex,tert,'-b')

ax.axis('equal')
plt.savefig('blade_shape_2D.pdf')