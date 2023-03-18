"""Given new blade sections and annulus lines, warp an existing grid."""

import numpy as np
from . import grid
import scipy.interpolate
import scipy.spatial


def warp(old_hdf5, ps, ss):

    # The jobs to do
    #   1. For each grid point on old surface, match percentage arc length to
    #      find a location for it on the new surface
    #   2. Calculate distortion values at each point on surface
    #   3. Find normals to the old surface
    #   4. Evaluate distortion decaying linearly away from old surface
    #   5. Move volume grid points according to distortion field

    # Read old grid and pull out blade surfaces
    g = grid.read_hdf5(old_hdf5)
    Cold = g.cut_blade_surfs(normalise=True)

    # Make the new PS/SS sections into one surface
    Cnew = []
    for ps_i, ss_i in zip(ps, ss):
        ps_now = np.array(ps_i)
        ss_now = np.array(ss_i)

        xrt_te = np.mean(np.stack((ps_now[:,-1,:],ss_now[:,-1,:])),axis=0)
        ps_now = np.concatenate((ps_now,np.expand_dims(xrt_te,1)),axis=1)
        ss_now = np.concatenate((ss_now,np.expand_dims(xrt_te,1)),axis=1)
        xrt_now = np.concatenate((np.flip(ss_now,axis=1),ps_now),axis=1)
        Cnew.append(xrt_now)

    # Locate the old points on new surface
    Csurf = []
    delta = []
    for cold, cnew in zip(Cold, Cnew):

        # Interpolate new points to old span fractions
        cnew_spf = np.mean((cnew[:,:,1] - cnew[:,:,1].min(axis=0, keepdims=True))/cnew[:,:,1].ptp(axis=0, keepdims=True),axis=-1)
        fx = scipy.interpolate.interp1d(cnew_spf,cnew[:,:,0],axis=0,kind='cubic')
        frt = scipy.interpolate.interp1d(cnew_spf,cnew[:,:,2],axis=0,kind='cubic')
        x_new = fx(np.mean(cold.spf,axis=0))
        rt_new = frt(cold.spf.mean(axis=0))

        zeta_new = grid._calculate_zeta(x_new.T, rt_new.T,normalise=True).T

        x_now = np.empty_like(cold.x)
        rt_now = np.empty_like(cold.x)
        for j in range(x_now.shape[1]):
            # print(x_new[j,:].min(),x_new[j,:].max())
            x_now[:,j] = np.interp(cold.zeta[:,j], zeta_new[j,:], x_new[j,:])
            rt_now[:,j] = np.interp(cold.zeta[:,j], zeta_new[j,:], rt_new[j,:])
        xrt_now = np.stack((x_now, cold.r, rt_now),axis=-1)
        dxrt_now = xrt_now - np.stack((cold.x, cold.r, cold.rt),axis=-1)
        Csurf.append(xrt_now)
        delta.append(dxrt_now)

    # Now we have a deformation vector for all points on the old surface
    # Assign maximum deformation by nearest neighbour search
    for cold, delt,row_bids in zip(Cold, delta, g.row_bids):

        cxref = cold.x.ptp()
        max_dist = cxref * 0.295
        coords = np.stack((cold.x.reshape(-1), cold.r.reshape(-1), cold.rt.reshape(-1)),axis=-1)
        deformation = np.stack([delt[:,:,i].reshape(-1) for i in range(3)], axis=-1)
        kdtree = scipy.spatial.cKDTree(coords.astype(np.float64))

        # Extract query points
        xrt_query = [g.unstructured_block_coords(bid) for bid in g.bids]

        # Function to query one block
        def _query_tree(xrtq):

            # Make the query
            dq, iq = kdtree.query(
                    xrtq.astype(np.float64),
                    distance_upper_bound=max_dist.astype(np.float64)
            )

            # Remove missing neighbour indexes
            N = kdtree.n
            dq[iq==N] = max_dist
            iq[iq==N] = 0.

            # Decay deformation away from surface
            del_q = deformation[iq]* np.expand_dims(1.-(dq/max_dist)**2.,1)

            return xrtq + del_q

        # Move the coords and set back to the grid
        xrt_moved = map(_query_tree, xrt_query)
        for xrtm, bid in zip(xrt_moved, g.bids):
            g.restructure_block_coords(bid, xrtm)

    return g

