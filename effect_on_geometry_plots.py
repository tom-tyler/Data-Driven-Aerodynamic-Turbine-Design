from turbine_design import turbine,turbine_GPR
import matplotlib.pyplot as plt

# model = turbine_GPR('eta_lost_5D')

# model.plot('phi',
#            'psi',
#            constants={'M2':0.7,
#                       'Co':0.65,
#                       'Lambda':0.5},
#            num_points=500,
#            contour_step=0.01,
#            fix_eta_lost_colors=True)

# fig,ax = plt.subplots(1,1)

turb = turbine(0.6,1.3,0.7,0.65)

turb.get_blade(2,
                        stack=False,
                        col='crimson')

turb = turbine(0.6,1.9,0.7,0.65)

turb.get_blade(2,
                        stack=False,
                        col='seagreen')

turb = turbine(1.0,1.3,0.7,0.65)

turb.get_blade(2,
                        stack=False,
                        col='cornflowerblue')

turb = turbine(1.0,1.9,0.7,0.65)

turb.get_blade(2,
                        stack=False,
                        col='darkorange')


# fig.add_axes(ax1A,label='A')#'$\\phi=0.6, \\psi=1.3$')
# fig.add_axes(ax1B,label='B')#'$\\phi=0.6, \\psi=1.9$')
# fig.add_axes(ax1C,label='C')#'$\\phi=1.0, \\psi=1.3$')
# fig.add_axes(ax1D,label='D')#'$\\phi=1.0, \\psi=1.9$')

# ax.set_title('$M_2=0.70$,   $\\Lambda=0.50$,   $C_0=0.65$',size=12)
# ax.grid(linestyle = '--', linewidth = 0.5)

# # plt.legend()

# leg = ax.legend(['(a)','(b)','(c)','(d)'])

# leg.legendHandles[0].set_color('seagreen')
# leg.legendHandles[1].set_color('darkorange')
# leg.legendHandles[2].set_color('crimson')
# leg.legendHandles[3].set_color('cornflowerblue')
      
# leg.set_draggable(state=True)

# plt.show()
