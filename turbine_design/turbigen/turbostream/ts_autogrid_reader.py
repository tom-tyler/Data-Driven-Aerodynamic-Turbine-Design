"""Turbostream Autogrid Reader modified to use double for intermediate calcs."""
from ts import ts_tstream_grid, ts_tstream_type, ts_tstream_default
from ts import ts_tstream_patch_kind, ts_tstream_check_grid
import numpy, math
import warnings


class AutogridReader:
    def __init__(self):
        self.block_names = {}
        self.g = ts_tstream_grid.TstreamGrid()
        self.patch_kind_map = {
            "PER": ts_tstream_patch_kind.periodic,
            "CON": ts_tstream_patch_kind.periodic,
            "INL": ts_tstream_patch_kind.inlet,
            "OUT": ts_tstream_patch_kind.outlet,
            "NMB": ts_tstream_patch_kind.nomatch,
            "ROT": ts_tstream_patch_kind.mixing,
            "PERNB": ts_tstream_patch_kind.nomatch,
            "EXT": ts_tstream_patch_kind.freestream,
        }

        self.iface_map = {0: 0, 1: 4, 2: 3, 3: 1, 4: 5, 5: 2}

    def join_mixing_blocks(self, g, bid0, bid1, bid2):
        b0 = g.get_block(bid0)
        b1 = g.get_block(bid1)
        b2 = g.get_block(bid2)

        x0 = g.get_bp("x", bid0)
        r0 = g.get_bp("r", bid0)
        rt0 = g.get_bp("rt", bid0)

        x1 = g.get_bp("x", bid1)
        r1 = g.get_bp("r", bid1)
        rt1 = g.get_bp("rt", bid1)

        # create the new block
        b3 = ts_tstream_type.TstreamBlock()
        ni3 = b0.ni
        nj3 = b0.nj
        nk3 = b0.nk + b1.nk - 1

        x3 = numpy.zeros((nk3, nj3, ni3), numpy.float64)
        r3 = numpy.zeros((nk3, nj3, ni3), numpy.float64)
        rt3 = numpy.zeros((nk3, nj3, ni3), numpy.float64)

        x3[0 : b0.nk, :, :] = x0
        r3[0 : b0.nk, :, :] = r0
        rt3[0 : b0.nk, :, :] = rt0

        x3[b0.nk - 1 :, :, :] = x1
        r3[b0.nk - 1 :, :, :] = r1
        rt3[b0.nk - 1 :, :, :] = rt1

        b3.ni = ni3
        b3.nj = nj3
        b3.nk = nk3

        bid3 = g.add_block(b3)
        b3.bid = bid3
        g.set_bp(
            "x",
            ts_tstream_type.float,
            bid3,
            numpy.zeros((nk3, nj3, ni3), numpy.float64) + x3,
        )
        g.set_bp(
            "r",
            ts_tstream_type.float,
            bid3,
            numpy.zeros((nk3, nj3, ni3), numpy.float64) + r3,
        )
        g.set_bp(
            "rt",
            ts_tstream_type.float,
            bid3,
            numpy.zeros((nk3, nj3, ni3), numpy.float64) + rt3,
        )

        p3 = ts_tstream_type.TstreamPatch()
        p3.kind = ts_tstream_patch_kind.mixing
        p3.bid = bid3
        p3.ist = b3.ni - 1
        p3.ien = b3.ni
        p3.jst = 0
        p3.jen = b3.nj
        p3.kst = 0
        p3.ken = b3.nk
        p3.nxbid = bid2
        p3.pid = g.add_patch(bid3, p3)

        for pid in g.get_patch_ids(bid2):
            p = g.get_patch(bid2, pid)
            if p.kind == ts_tstream_patch_kind.mixing:
                p.nxbid = p3.bid
                p.nxpid = p3.pid
                p3.nxpid = pid

        # add patches on b0 to b3
        for pid in g.get_patch_ids(bid0):
            p0 = g.get_patch(bid0, pid)
            if (
                not (p0.nxbid == bid1 and p0.kst == b0.nk - 1)
            ) and p0.kind == ts_tstream_patch_kind.periodic:
                p3 = ts_tstream_type.TstreamPatch()
                p3.kind = ts_tstream_patch_kind.periodic
                p3.bid = bid3
                p3.ist = p0.ist
                p3.ien = p0.ien
                p3.jst = p0.jst
                p3.jen = p0.jen
                p3.kst = p0.kst
                p3.ken = p0.ken
                p3.nxbid = p0.nxbid
                p3.nxpid = p0.nxpid
                p3.idir = p0.idir
                p3.jdir = p0.jdir
                p3.kdir = p0.kdir
                p3.pid = g.add_patch(bid3, p3)

                nxp = g.get_patch(p3.nxbid, p3.nxpid)
                nxp.nxbid = p3.bid
                nxp.nxpid = p3.pid

        # add patches on b1 to b3
        for pid in g.get_patch_ids(bid1):
            p1 = g.get_patch(bid1, pid)
            if (
                not (p1.nxbid == bid0 and p1.kst == 0)
            ) and p1.kind == ts_tstream_patch_kind.periodic:
                p3 = ts_tstream_type.TstreamPatch()
                p3.kind = ts_tstream_patch_kind.periodic
                p3.bid = bid3
                p3.ist = p1.ist
                p3.ien = p1.ien
                p3.jst = p1.jst
                p3.jen = p1.jen
                p3.kst = p1.kst + b0.nk - 1
                p3.ken = p1.ken + b0.nk - 1
                p3.nxbid = p1.nxbid
                p3.nxpid = p1.nxpid
                p3.idir = p1.idir
                p3.jdir = p1.jdir
                p3.kdir = p1.kdir
                p3.pid = g.add_patch(bid3, p3)

                nxp = g.get_patch(p3.nxbid, p3.nxpid)
                nxp.nxbid = p3.bid
                nxp.nxpid = p3.pid

        for bvid in g.get_bv_ids():
            g.set_bv(bvid, g.get_bv_type(bvid), bid3, g.get_bv(bvid, bid0))

        # remove b0 and b1
        g.remove_block_and_update(bid0)
        g.remove_block_and_update(bid1)

    def read_patches(self, f, bid, sliding):
        g = self.g
        b = g.get_block(bid)

        l = f.readline().split()
        self.block_names[l[0]] = bid

        l = f.readline()

        l = f.readline().split()

        # JB: not sure when this would fail
        # try:
        #     nblade = int(l[-2])
        # except:
        #     nblade = 1

        nblade = int(l[-2])

        g.set_bv("nblade", ts_tstream_type.int, bid, nblade)
        g.set_bv("fblade", ts_tstream_type.float, bid, nblade)

        for iface in range(6):  # [0, 4, 3, 1, 5, 2]:
            iface = self.iface_map[iface]

            l = f.readline().split()
            np = int(l[0])
            for ipatch in range(np):
                l = f.readline().split()
                label = l[0]
                type = l[1]

                if type in [
                    "INL",
                    "OUT",
                    "EXT",
                    "PER",
                    "CON",
                    "ROT",
                    "NMB",
                    "PERNB",
                ]:
                    st0, st1, en0, en1 = [int(tmp) for tmp in l[2:6]]

                if type in ["PER", "CON", "NMB", "PERNB"]:
                    nxbid = int(l[6]) - 1
                    nxiface = self.iface_map[int(l[7]) - 1]
                    nxipatch = int(l[8]) - 1
                    ori = int(l[9])
                    cor = int(l[10])

                if type in [
                    "PER",
                    "CON",
                    "INL",
                    "EXT",
                    "OUT",
                    "NMB",
                    "ROT",
                    "PERNB",
                ]:
                    p = ts_tstream_type.TstreamPatch()
                    p.bid = bid
                    p.kind = self.patch_kind_map[type]

                    if iface in [0, 2]:
                        if iface == 0:
                            p.ist = 0
                            p.ien = 1
                        if iface == 2:
                            p.ist = b.ni - 1
                            p.ien = b.ni

                        p.jst = st0 - 1
                        p.jen = en0
                        p.kst = st1 - 1
                        p.ken = en1

                    if iface in [4, 5]:
                        if iface == 4:
                            p.jst = 0
                            p.jen = 1
                        if iface == 5:
                            p.jst = b.nj - 1
                            p.jen = b.nj

                        p.ist = st0 - 1
                        p.ien = en0
                        p.kst = st1 - 1
                        p.ken = en1

                    if iface in [1, 3]:
                        if iface == 3:
                            p.kst = 0
                            p.ken = 1
                        if iface == 1:
                            p.kst = b.nk - 1
                            p.ken = b.nk

                        p.ist = st0 - 1
                        p.ien = en0
                        p.jst = st1 - 1
                        p.jen = en1

                    p.pid = g.add_patch(bid, p)

                if type in ["PER", "CON", "NMB", "PERNB"]:
                    p.nxbid = nxbid
                    # store connection and orientation for later
                    p.iface = iface
                    p.ipatch = ipatch
                    p.nxiface = nxiface
                    p.nxipatch = nxipatch
                    p.ori = ori
                    p.cor = cor

                    self.patch_face_map[bid][iface][ipatch] = p

                if type in ["ROT"] and sliding == False:
                    p.iface = iface
                    p.ipatch = ipatch
                    p.label = label.split("(")[1][0:2]
                    self.patch_face_map[bid][iface][ipatch] = p
                    self.mixing_plane_map[bid][iface] = p

    def swap(self, a, b):
        tmp = a
        a = b
        b = tmp
        return a, b

    def connect_patches(self, sliding):
        g = self.g
        for bid in g.get_block_ids():
            b = g.get_block(bid)
            for pid in g.get_patch_ids(bid):
                p = g.get_patch(bid, pid)

                if (
                    p.kind in [ts_tstream_patch_kind.mixing]
                    and sliding == False
                ):

                    # find the matching mixing plane
                    nxp_keep = None

                    for nxbid in self.mixing_plane_map:
                        for nxiface in self.mixing_plane_map[nxbid]:

                            nxp = self.mixing_plane_map[nxbid][nxiface]

                            if (p.label == nxp.label) and (p.bid, p.iface) != (
                                nxbid,
                                nxiface,
                            ):

                                nxp_keep = nxp

                    p.nxbid = nxp_keep.bid
                    p.nxpid = nxp_keep.pid

                if p.kind in [
                    ts_tstream_patch_kind.periodic,
                    ts_tstream_patch_kind.nomatch,
                ]:

                    nxp = self.patch_face_map[p.nxbid][p.nxiface][p.nxipatch]
                    p.nxpid = nxp.pid

                    if p.iface in [0, 2]:
                        dir0 = 1
                        dir1 = 2
                    if p.iface in [4, 5]:
                        dir0 = 0
                        dir1 = 2
                    if p.iface in [1, 3]:
                        dir0 = 0
                        dir1 = 1

                    if p.nxiface in [0, 2]:
                        nxdir0 = 1
                        nxdir1 = 2
                    if p.nxiface in [4, 5]:
                        nxdir0 = 0
                        nxdir1 = 2
                    if p.nxiface in [1, 3]:
                        nxdir0 = 0
                        nxdir1 = 1

                    # swap axis
                    if (p.ori, p.cor) in [(0, 1), (0, 3), (1, 0), (1, 2)]:
                        nxdir0, nxdir1 = self.swap(nxdir0, nxdir1)

                    # reverse dir0
                    if (p.ori, p.cor) in [(0, 2), (0, 3), (1, 1), (1, 2)]:
                        nxdir0 += 3

                    # reverse dir1
                    if (p.ori, p.cor) in [(0, 1), (0, 2), (1, 2), (1, 3)]:
                        nxdir1 += 3

                    nxdir = [6, 6, 6]

                    nxdir[dir0] = nxdir0
                    nxdir[dir1] = nxdir1

                    p.idir, p.jdir, p.kdir = nxdir[0], nxdir[1], nxdir[2]

    def read(self, bcs_path, p3d_path, sliding=False):
        g = self.g

        print "AutogridReader: Reading %s" % (p3d_path)
        f = open(p3d_path, "r")
        nb = int(f.readline())
        l_n = []
        for ib in range(nb):
            l_n += [int(s) for s in f.readline().split()]

        l = [float(s) for s in f.read().split()]
        i000 = 0
        for ib in range(nb):
            print "\tReading block no. %i" % (ib)
            ni, nj, nk = l_n[3 * ib : 3 * ib + 3]

            b = ts_tstream_type.TstreamBlock()
            b.bid = ib
            b.np = 0
            b.ni = ni
            b.nj = nj
            b.nk = nk
            b.procid = 0
            b.threadid = 0
            g.add_block(b)

            x = numpy.zeros((nk, nj, ni), numpy.float64)
            y = numpy.zeros((nk, nj, ni), numpy.float64)
            z = numpy.zeros((nk, nj, ni), numpy.float64)

            ntot = ni * nj * nk
            for k in range(nk):
                for j in range(nj):
                    for i in range(ni):
                        x[k, j, i] = l[i000]
                        y[k, j, i] = l[i000 + ntot]
                        z[k, j, i] = l[i000 + 2 * ntot]
                        i000 += 1
            i000 += 2 * ntot

            g.set_bp("x", ts_tstream_type.float, ib, x)
            g.set_bp("r", ts_tstream_type.float, ib, y)
            g.set_bp("rt", ts_tstream_type.float, ib, z)
        f.close()

        self.mixing_plane_map = {}
        self.patch_face_map = {}
        for bid in g.get_block_ids():
            self.patch_face_map[bid] = {}
            self.mixing_plane_map[bid] = {}
            for iface in range(6):
                self.patch_face_map[bid][iface] = {}

        print "AutogridReader: Reading %s" % (bcs_path)
        f = open(bcs_path, "r")
        f.readline()
        for bid in g.get_block_ids():
            self.read_patches(f, bid, sliding)

        # JB: replace commented out rpms section with this
        rpms = []
        for l in f.readlines():
            property_name, property_val = l.split()[:2]
            if property_name == "ROTATION_SPEED":
                rpms.append(float(property_val))
        for rpm, bid in zip(rpms, g.get_block_ids()):
            g.set_bv("rpm", ts_tstream_type.float, bid, rpm)

        f.close()

        self.connect_patches(sliding)

        for bid in g.get_block_ids():
            b = g.get_block(bid)
            x = g.get_bp("x", bid)
            y = g.get_bp("r", bid)
            z = g.get_bp("rt", bid)

            # swap i and k
            ni = b.ni
            nj = b.nj
            nk = b.nk

            ni2 = nk
            nj2 = nj
            nk2 = ni

            x2 = numpy.zeros((nk2, nj2, ni2), numpy.float64)
            y2 = numpy.zeros((nk2, nj2, ni2), numpy.float64)
            z2 = numpy.zeros((nk2, nj2, ni2), numpy.float64)

            for j in range(nj):
                for k in range(nk):
                    for i in range(ni):
                        i2 = k
                        j2 = j
                        k2 = i

                        x2[k2, j2, i2] = x[k, j, i]
                        y2[k2, j2, i2] = y[k, j, i]
                        z2[k2, j2, i2] = z[k, j, i]

            b.ni, b.nj, b.nk = ni2, nj2, nk2

            # reverse i
            x3 = numpy.zeros((nk2, nj2, ni2), numpy.float64)
            y3 = numpy.zeros((nk2, nj2, ni2), numpy.float64)
            z3 = numpy.zeros((nk2, nj2, ni2), numpy.float64)
            for j in range(nj2):
                for k in range(nk2):
                    for i in range(ni2):
                        i2 = ni2 - i - 1
                        j2 = j
                        k2 = k  # nk2 - k - 1

                        x3[k2, j2, i2] = x2[k, j, i]
                        y3[k2, j2, i2] = y2[k, j, i]
                        z3[k2, j2, i2] = z2[k, j, i]

            # convert to cylindrical polar
            x4 = z3
            r4 = numpy.sqrt(y3 * y3 + x3 * x3)
            np = numpy
            # rt4 = r4*numpy.arctan2(y3, x3)
            t4 = numpy.arctan2(y3, x3)

            # check angle in i-dir
            for j in range(nj2):
                for k in range(nk2):
                    for i in range(0, ni2 - 1):
                        dangle = t4[k, j, i + 1] - t4[k, j, i]
                        if abs(dangle) > math.pi:
                            print "AutogridReader: Changing angles in the i-dir for bid=%i i=%i and j=%i" % (
                                bid,
                                i,
                                j,
                            )
                            isign = -1
                            if dangle < 1:
                                isign = 1

                            for ii in range(i + 1, ni2):
                                t4[k, j, ii] = (
                                    t4[k, j, ii] + 2 * math.pi * isign
                                )

            # check angle in j-dir
            for i in range(ni2):
                for k in range(nk2):
                    for j in range(0, nj2 - 1):
                        dangle = t4[k, j + 1, i] - t4[k, j, i]
                        if abs(dangle) > math.pi:
                            print "AutogridReader: Changing angles in the j-dir for bid=%i i=%i and j=%i" % (
                                bid,
                                i,
                                j,
                            )
                            isign = -1
                            if dangle < 1:
                                isign = 1

                            for jj in range(j + 1, nj2):
                                t4[k, jj, i] = (
                                    t4[k, jj, i] + 2 * math.pi * isign
                                )

            # check angle in the k-dir
            for j in range(nj2):
                for i in range(ni2):
                    for k in range(0, nk2 - 1):
                        dangle = t4[k + 1, j, i] - t4[k, j, i]
                        if abs(dangle) > math.pi:
                            print "AutogridReader: Changing angles in the k-dir for bid=%i i=%i and j=%i" % (
                                bid,
                                i,
                                j,
                            )
                            isign = -1
                            if dangle < 1:
                                isign = 1

                            for kk in range(k + 1, nk2):
                                t4[kk, j, i] = (
                                    t4[kk, j, i] + 2 * math.pi * isign
                                )
            rt4 = r4 * t4

            g.set_bp("x", ts_tstream_type.float, bid, x4.astype(numpy.float32))
            g.set_bp("r", ts_tstream_type.float, bid, r4.astype(numpy.float32))
            g.set_bp(
                "rt", ts_tstream_type.float, bid, rt4.astype(numpy.float32)
            )

        # update patches for k-i swap
        for bid in g.get_block_ids():
            b = g.get_block(bid)

            for pid in g.get_patch_ids(bid):
                p = g.get_patch(bid, pid)

                ist = p.ist
                ien = p.ien
                kst = p.kst
                ken = p.ken

                p.ist = kst
                p.ien = ken
                p.kst = ist
                p.ken = ien

                if p.kind in [
                    ts_tstream_patch_kind.periodic,
                    ts_tstream_patch_kind.nomatch,
                ]:

                    idir = p.idir
                    kdir = p.kdir

                    dirmap = {
                        6: 6,
                        0: 2,  # +i -> +k
                        1: 1,  # +j -> +j
                        2: 0,  # +k -> +i
                        3: 5,  # -i -> -k,
                        4: 4,  # -j -> -j,
                        5: 3,
                    }  # -k -> -i

                    p.idir = kdir
                    p.kdir = idir

                    p.idir = dirmap[p.idir]
                    p.jdir = dirmap[p.jdir]
                    p.kdir = dirmap[p.kdir]

        # update patches for i-reversal
        for bid in g.get_block_ids():
            b = g.get_block(bid)
            for pid in g.get_patch_ids(bid):
                p = g.get_patch(bid, pid)

                ist = p.ist
                ien = p.ien
                p.ist = b.ni - ien
                p.ien = b.ni - ist

                if p.kind in [
                    ts_tstream_patch_kind.periodic,
                    ts_tstream_patch_kind.nomatch,
                ]:

                    idirmap = {
                        6: 6,
                        0: 0,  # +i -> +i
                        1: 4,  # +j -> -j
                        2: 5,  # +k -> -k
                        3: 3,  # -i -> -i,
                        4: 1,  # -j -> +j,
                        5: 2,
                    }  # -k -> +k

                    jkdirmap = {
                        6: 6,
                        0: 3,  # +i -> -i
                        1: 1,  # +j -> +j
                        2: 2,  # +k -> +k
                        3: 0,  # -i -> +i,
                        4: 4,  # -j -> -j,
                        5: 5,
                    }  # -k -> -k

                    p.idir = idirmap[p.idir]
                    p.jdir = jkdirmap[p.jdir]
                    p.kdir = jkdirmap[p.kdir]

        for name in ts_tstream_default.av:
            if name not in g.get_av_ids():
                val = ts_tstream_default.av[name]
                if type(val) == type(1):
                    g.set_av(name, ts_tstream_type.int, val)
                else:
                    g.set_av(name, ts_tstream_type.float, val)

        for name in ts_tstream_default.bv:
            if name not in g.get_bv_ids():
                for bid in g.get_block_ids():
                    val = ts_tstream_default.bv[name]
                    if type(val) == type(1):
                        g.set_bv(name, ts_tstream_type.int, bid, val)
                    else:
                        g.set_bv(name, ts_tstream_type.float, bid, val)

        return g
