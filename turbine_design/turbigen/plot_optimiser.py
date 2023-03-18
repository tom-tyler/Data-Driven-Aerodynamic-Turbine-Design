import openmdao.api as om
import numpy as np
import matplotlib.pyplot as plt
import os


def _input_to_x(d):
    x = []
    lab = []
    for k in d:
        for v in d[k]:
            x.append(v)
            lab.append(k)
    return np.array(x), lab


class OptimisePlotter:
    def __init__(self, sql_file, prefix=None):
        """Read a recorder database and make plots."""

        # Load data
        cr = om.CaseReader(sql_file)

        # Extract list of cases
        self.cases = cr.get_cases("driver", recurse=False)
        self.basedir = os.path.dirname(sql_file)

        # Store vars
        self.eta_lost = self._get_var("lost_efficiency_percent")
        self.recamber_te = self._get_var("recamber_te_norm")
        self.stagger = self._get_var("stagger_norm")
        self.beta = self._get_var("beta_norm")
        self.Rle = self._get_var("radius_le_norm")
        self.thickness = self._get_var("thickness_norm")
        self.xtmax = self._get_var("xtmax_norm")

        # Constraint errors
        self.err_Lam = self._get_var("err_Lam_rel")
        self.err_psi = self._get_var("err_psi_rel")
        self.err_phi = self._get_var("err_phi_rel")

        self.prefix = prefix if prefix else ""

    def _get_var(self, k):
        return np.array([c[k] for c in self.cases])

    def steps(self):
        """Plot the change in efficiency due to initial steps of each var."""

        x0, lab0 = _input_to_x(self.cases[0].inputs)
        nv = len(x0)

        x = np.stack([_input_to_x(c.inputs)[0] for c in self.cases[1:nv]])

        # Calc changes for original simplex
        deta = np.diff(self.eta_lost[1:nv, 0])
        dx = np.diff(x, n=1, axis=0)

        lab = np.take(lab0, np.argmax(dx, axis=1))
        lab_format = [l[:-5] for l in lab]

        fig, ax = plt.subplots()
        hpos = np.linspace(0, nv - 3, nv - 2)
        ax.bar(hpos, deta)
        ax.set_xticks(hpos)
        ax.set_xticklabels(lab_format, rotation=90.0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.basedir, "sens%s.pdf" % self.prefix))

    def convergence(self):

        fig, ax = plt.subplots()
        ax.plot(self.eta_lost, "-")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Lost efficiency, $\Delta \eta / \%$")
        plt.tight_layout()
        plt.savefig(os.path.join(self.basedir, "conv%s.pdf" % self.prefix))

    def constraints(self):

        fig, ax = plt.subplots()
        ax.plot(self.err_phi)
        ax.plot(self.err_psi)
        ax.plot(self.err_Lam)
        ax.axhline(-1.0, c="k", linestyle="--")
        ax.axhline(1.0, c="k", linestyle="--")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Constraint Error")
        plt.tight_layout()
        plt.savefig(os.path.join(self.basedir, "err%s.pdf" % self.prefix))

    def all(self):
        self.convergence()
        self.constraints()
        self.steps()
