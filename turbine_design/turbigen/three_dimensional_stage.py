"""Parameters for three-dimensional turbine stage design"""
import json
import numpy as np
from . import mean_line_stage, geometry
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


def _fit_tmax_xtmax(A_target):

    A_target = A_target.reshape(-1)

    def _iter(xq, t, xt):
        x = np.array([0.0, xt, 1.0])
        A = np.insert(
            A_target[
                (0, -1),
            ],
            1,
            t,
        )
        return interp1d(x, A, kind="linear")(xq)

    xi = np.linspace(0.0, 1.0, len(A_target))
    return curve_fit(_iter, xi, A_target, (0.3, 0.5))[0]


class RowParameterSet:
    """Encapsulate the set of parameters needed to make a blade section."""

    def __init__(self, row_dict):

        self.tte = row_dict.pop("tte")
        self.spf = np.array([row_dict[k]["spf"] for k in row_dict])
        # Sort everything by spf
        jsort = np.argsort(self.spf)
        self.spf = self.spf[jsort]
        self.stagger = np.array([row_dict[k]["stagger"] for k in row_dict])[
            jsort
        ]
        self.recamber = np.stack([row_dict[k]["recamber"] for k in row_dict])[
            jsort
        ]
        self.Rle = np.array([row_dict[k]["Rle"] for k in row_dict])[jsort]
        self.beta = np.array([row_dict[k]["beta"] for k in row_dict])[jsort]
        self.thickness_ps = np.array(
            [row_dict[k]["thickness_ps"] for k in row_dict]
        )[jsort]
        self.thickness_ss = np.array(
            [row_dict[k]["thickness_ss"] for k in row_dict]
        )[jsort]
        self.max_thickness_location_ss = np.array(
            [row_dict[k]["max_thickness_location_ss"] for k in row_dict]
        )[jsort]
        self.max_thickness_location_ps = np.array(
            [row_dict[k]["max_thickness_location_ps"] for k in row_dict]
        )[jsort]
        self.lean = np.array([row_dict[k]["lean"] for k in row_dict])[jsort]
        self.nsect = len(self.spf)

        # Disable new attribute creation from now on to catch silent typos
        # self._freeze()

    def to_dict(self):
        row_dict = {}
        row_dict["tte"] = self.tte
        for k in range(self.nsect):
            kstr = "sect_%d" % k
            row_dict[kstr] = {
                "spf": self.spf[k],
                "stagger": self.stagger[k],
                "recamber": self.recamber[k].tolist(),
                "Rle": self.Rle[k],
                "beta": self.beta[k],
                "thickness_ps": self.thickness_ps[k],
                "thickness_ss": self.thickness_ss[k],
                "max_thickness_location_ss": self.max_thickness_location_ss[k],
                "max_thickness_location_ps": self.max_thickness_location_ps[k],
                "lean": self.lean[k],
            }
        return row_dict

    @property
    def A(self):

        # Convert Rle, beta to A coefficients
        Ale = np.sqrt(2.0 * self.Rle).reshape(-1, 1)
        Ate = (np.tan(np.radians(self.beta)) + self.tte).reshape(-1, 1)

        # Between leading and trailing edge fill in a number of thickness
        # coefficients to describe the profile
        nA = 5

        # # Pressure side - constant thickness along chord
        # A_ps = np.tile(self.thickness_ps.reshape(-1, 1), (1, nA - 2))
        # A_ps = np.concatenate((Ale, A_ps, Ate), axis=1)

        # Loop over all sections
        A_ss = np.empty((self.nsect, nA))
        A_ps = np.empty((self.nsect, nA))
        xq = np.linspace(0.0, 1.0, nA)
        for isect in range(self.nsect):

            # On suction side (upper for vane) vary up to tmax at xmax linear
            x = np.array([0.0, self.max_thickness_location_ss[isect], 1.0])
            A = np.array(
                [Ale[isect, 0], self.thickness_ss[isect], Ate[isect, 0]]
            )
            A_ss[isect] = interp1d(x, A, kind="linear")(xq)

            x = np.array([0.0, self.max_thickness_location_ps[isect], 1.0])
            A = np.array(
                [Ale[isect, 0], self.thickness_ps[isect], Ate[isect, 0]]
            )
            A_ps[isect] = interp1d(x, A, kind="linear")(xq)

        return A_ps, A_ss

    @property
    def interp_method(self):
        if self.nsect == 1:
            return None
        elif self.nsect == 2:
            return "slinear"
        elif self.nsect == 3:
            return "quadratic"
        else:
            return "cubic"

    def interpolate_stagger(self, spf_q):
        if self.nsect > 1:
            func_stagger = interp1d(
                self.spf,
                np.tan(np.radians(self.stagger)),
                kind=self.interp_method,
            )
            return np.degrees(np.arctan(func_stagger(spf_q)))
        else:
            nr = 1 if np.isscalar(spf_q) else len(spf_q)
            return np.tile(self.stagger, (nr, 1))

    def interpolate_lean(self, spf_q):
        if self.nsect > 1:
            func_lean = interp1d(self.spf, self.lean, kind=self.interp_method)
            return func_lean(spf_q)
        else:
            # If we only have one section, then lean symmetrically
            spf = np.array([0.0, 0.5, 1.0])
            lean = np.array([0.0, self.lean[0], 0.0])
            func_lean = interp1d(spf, lean, kind="quadratic")
            return func_lean(spf_q)

    def interpolate_A(self, spf_q):
        # Interpolators
        if self.nsect > 1:
            A_upper, A_lower = self.A
            func_A_upper = interp1d(
                self.spf, A_upper, axis=0, kind=self.interp_method
            )
            func_A_lower = interp1d(
                self.spf, A_lower, axis=0, kind=self.interp_method
            )
            return func_A_upper(spf_q), func_A_lower(spf_q)
        else:
            nr = 1 if np.isscalar(spf_q) else len(spf_q)
            return np.tile(self.A, (nr, 1))

    def interpolate_recamber(self, spf_q):
        if self.nsect > 1:
            recam = np.atleast_2d(
                interp1d(self.spf, self.recamber, axis=0, kind="quadratic")(
                    spf_q
                )
            )
        else:
            nr = 1 if np.isscalar(spf_q) else len(spf_q)
            recam = np.tile(self.recamber, (nr, 1))
        return recam


class StageParameterSet:
    """Encapsulate the set of parameters sufficient to run a case."""

    def __init__(self, var_dict):
        """Create a parameter set using a dictionary."""

        # Assign dictionary items to the class

        # Mean-line
        self.phi = var_dict["mean-line"]["phi"]
        self.psi = var_dict["mean-line"]["psi"]
        self.Lam = var_dict["mean-line"]["Lam"]
        self.Al1 = var_dict["mean-line"]["Al1"]
        self.Ma2 = var_dict["mean-line"]["Ma2"]
        self.eta_guess = var_dict["mean-line"]["eta"]
        self.ga = var_dict["mean-line"]["ga"]
        self.loss_split = var_dict["mean-line"]["loss_split"]
        self.fc = np.array(var_dict["mean-line"]["fc"])
        self.TRc = np.array(var_dict["mean-line"]["TRc"])

        # Boundary conditions
        self.To1 = var_dict["bcond"]["To1"]
        self.Po1 = var_dict["bcond"]["Po1"]
        self.rgas = var_dict["bcond"]["rgas"]
        self.Omega = var_dict["bcond"]["Omega"]
        self.delta = var_dict["bcond"]["delta"]

        # Three-dimensional design parameters
        self.htr = var_dict["3d"]["htr"]
        self.Co = np.array(var_dict["3d"]["Co"])
        self.AR = np.array(var_dict["3d"]["AR"])
        self.tau_c = var_dict["3d"]["tau_c"]
        self.Re = var_dict["3d"]["Re"]

        # Meshing parameters
        self.dx_c = np.array(var_dict["mesh"]["dx_c"])

        # CFD running parameters
        self.guess_file = var_dict["run"]["guess_file"]
        self.grid_type = var_dict["run"]["grid_type"]
        self.cfd_config = var_dict["run"]["cfd_config"]

        # Determine how many rows have sections defined
        self.nrow_sections = np.sum([k.startswith("sect") for k in var_dict])

        # Get row parameters for these sections
        self.row_sections = [
            RowParameterSet(var_dict["sect_row_%d" % irow])
            for irow in range(self.nrow_sections)
        ]

        # Disable new attribute creation from now on to catch silent typos
        # self._freeze()

    @property
    def nondimensional_stage(self):
        return mean_line_stage.nondim_stage_from_Lam(
            self.phi,
            self.psi,
            self.Lam,
            self.Al1,
            self.Ma2,
            self.ga,
            self.eta_guess,
            loss_rat=self.loss_split,
            mdotc_mdot1=self.fc,
            Toc_Toinf=self.TRc,
        )

    @property
    def dimensional_stage(self):
        return mean_line_stage.scale_geometry(
            self.nondimensional_stage,
            self.htr,
            self.Omega,
            self.To1,
            self.Po1,
            self.rgas,
            self.Co,
            self.AR,
            self.Re,
        )

    @classmethod
    def from_json(cls, fname):
        """Create a parameter set from a file on disk."""

        # Load the data
        with open(fname, "r") as f:
            dat = json.load(f)

        # Pass dict to the normal init method
        return cls(dat)

    def free_vortex_chi(self, spf_q):
        """Twist the metal angles of vane/blade in free vortex, with recamber."""
        chi = np.stack(
            (
                self.dimensional_stage.free_vortex_vane(spf_q),
                self.dimensional_stage.free_vortex_blade(spf_q),
            )
        )
        recamber = np.stack(
            [row.interpolate_recamber(spf_q).T for row in self.row_sections]
        )
        # Invert recambering in rotor
        recamber[1] *= -1.0
        return chi + recamber

    def estimate_stagger(self, irow, spf_q):
        """Get stagger variation away from midspan from camber angles."""

        row_sections = self.row_sections[irow]

        # Evaluate interpolators at query span fractions
        chi_q = self.free_vortex_chi(spf_q)[irow].reshape(-1)

        # Datum stagger angle at midspan based on camber angles
        chi_mid = self.free_vortex_chi(0.5)[irow].reshape(-1)
        stag_datum_mid = np.degrees(
            np.arctan(np.mean(np.tan(np.radians(chi_mid))))
        ).reshape(-1)[0]

        # Actual stagger angle
        stag_mid = row_sections.interpolate_stagger(0.5).reshape(-1)[0]

        # Datum stagger angle at all radii
        stag_datum = np.degrees(np.arctan(np.mean(np.tan(np.radians(chi_q)))))

        # The query stagger is the datum stagger with offset wrt midspan
        stag_q = stag_datum + (stag_mid - stag_datum_mid)

        return stag_q

    def interpolate_section(self, irow, spf_q, is_rotor=False):
        """Get blade section coordinates for a given row and span location."""

        row_sections = self.row_sections[irow]

        # Evaluate interpolators at query span fractions
        chi_q = self.free_vortex_chi(spf_q)[irow].reshape(-1)

        if np.all(row_sections.stagger == 0.0):
            stag_q = np.degrees(np.arctan(np.mean(np.tan(np.radians(chi_q)))))

        elif row_sections.nsect == 1:

            stag_q = self.estimate_stagger(irow, spf_q)

        else:
            stag_q = row_sections.interpolate_stagger(spf_q).reshape(-1)

        A_q = row_sections.interpolate_A(spf_q)

        # The rotor suction side is opposite to vane suction side - flip
        if is_rotor:
            A_q = np.flip(A_q, axis=0)

        # Trailing edge thickness
        tte = row_sections.tte

        # Second, convert thickness in shape space to real coords
        sec_xrt = np.squeeze(geometry._section_xy(chi_q, A_q, tte, stag_q))

        # Put leading edge at x=0
        sec_xrt[:, 0, :] -= sec_xrt[:, 0, :].min()

        if is_rotor:
            # Make into a loop
            loop_xrt = geometry._loop_section(sec_xrt)
            # Get centroid rt coordinate
            cent_rt = geometry.centroid(loop_xrt)
            # Subtract from PS and SS to put centroid at t=0
            sec_xrt[:, 1, :] -= cent_rt

        else:
            # Lean the blades
            lean_q = row_sections.interpolate_lean(spf_q)
            sec_xrt[:, 1, :] += lean_q

        return sec_xrt

    def to_dict(self):
        """Dictionary with copies of data."""
        var_dict = {
            "mean-line": {},
            "bcond": {},
            "3d": {},
            "mesh": {},
            "run": {},
        }

        # Mean-line
        var_dict["mean-line"]["phi"] = float(self.phi)
        var_dict["mean-line"]["psi"] = float(self.psi)
        var_dict["mean-line"]["Lam"] = float(self.Lam)
        var_dict["mean-line"]["Al1"] = float(self.Al1)
        var_dict["mean-line"]["Ma2"] = float(self.Ma2)
        var_dict["mean-line"]["eta"] = float(self.eta_guess)
        var_dict["mean-line"]["ga"] = float(self.ga)
        var_dict["mean-line"]["loss_split"] = float(self.loss_split)
        var_dict["mean-line"]["fc"] = self.fc.tolist()
        var_dict["mean-line"]["TRc"] = self.TRc.tolist()

        # Boundary conditions
        var_dict["bcond"]["To1"] = float(self.To1)
        var_dict["bcond"]["Po1"] = float(self.Po1)
        var_dict["bcond"]["rgas"] = float(self.rgas)
        var_dict["bcond"]["Omega"] = float(self.Omega)
        var_dict["bcond"]["delta"] = float(self.delta)

        # Three-dimensional design parameters
        var_dict["3d"]["htr"] = float(self.htr)
        var_dict["3d"]["Co"] = self.Co.tolist()
        var_dict["3d"]["tau_c"] = float(self.tau_c)
        var_dict["3d"]["AR"] = self.AR.tolist()
        var_dict["3d"]["Re"] = float(self.Re)

        # Meshing parameters
        var_dict["mesh"]["dx_c"] = self.dx_c.tolist()

        # CFD running parameters
        var_dict["run"]["guess_file"] = self.guess_file
        var_dict["run"]["grid_type"] = self.grid_type
        var_dict["run"]["cfd_config"] = self.cfd_config.copy()

        # Do blade sections
        for irow in range(self.nrow_sections):
            var_dict["sect_row_%d" % irow] = self.row_sections[irow].to_dict()

        return var_dict

    def to_json(self, fname):
        """Write this parameter set to a JSON file."""
        with open(fname, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    def copy(self):
        """Return a copy of this parameter set."""
        return StageParameterSet(self.to_dict())

    def refine_sections(self, spf_q):
        """Replace existing blade sections with more new sections at spf_q."""

        # Loop over rows
        row_sections_new = []
        for irow in range(2):

            row_sections_old = self.row_sections[irow]
            row_dict = {"tte": row_sections_old.tte}

            # Loop over new span fractions
            for j, spf in enumerate(spf_q):

                Au, Al = row_sections_old.interpolate_A(spf)
                Rle = 0.5 * Au[0, 0] ** 2.0
                beta = np.degrees(np.arctan(Au[0, -1] - row_sections_old.tte))

                tmaxu, xtmaxu = _fit_tmax_xtmax(Au)
                tmaxl, xtmaxl = _fit_tmax_xtmax(Al)

                sect_dict = {
                    "spf": spf,
                    "stagger": self.estimate_stagger(irow, spf),
                    "recamber": np.squeeze(
                        row_sections_old.interpolate_recamber(spf)
                    ).tolist(),
                    "Rle": Rle,
                    "beta": beta,
                    "thickness_ps": tmaxu,
                    "thickness_ss": tmaxl,
                    "max_thickness_location_ps": xtmaxu,
                    "max_thickness_location_ss": xtmaxl,
                    "lean": row_sections_old.interpolate_lean(spf).tolist(),
                }

                kstr = "sect_%d" % j
                row_dict[kstr] = sect_dict

            row_sections_new.append(RowParameterSet(row_dict))

        self.row_sections = row_sections_new
