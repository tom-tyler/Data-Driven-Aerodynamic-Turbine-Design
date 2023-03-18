"""Use OpenMDAO to run optimisations on three-dimensional turbines."""
import openmdao.api as om
from . import util, three_dimensional_stage
import os, json
import numpy as np

LOG_SQL = "opt.sql"

INPUT_LOWER = {
    "thickness": 0.2,
    "stagger": -90.0,
    "beta": 10.0,
    "radius_le": 0.04,
    "recamber_le": -10.0,
    "recamber_te": -10.0,
    "lean": 0.0,
    "xtmax": 0.1,
}

INPUT_UPPER = {
    "thickness": 0.6,
    "stagger": 90.0,
    "beta": 30.0,
    "radius_le": 0.15,
    "recamber_le": 10.0,
    "recamber_te": 10.0,
    "lean": 0.5,
    "xtmax": 0.9,
}

INPUT_REFERENCE = {
    "thickness": 0.3,
    "stagger": 45.0,
    "beta": 10.0,
    "radius_le": 0.05,
    "recamber_le": 0.0,
    "recamber_te": 0.0,
    "lean": 0.0,
    "xtmax": 0.5,
}

INPUT_SENSITIVITY = {
    "stagger": 2.0,
    "beta": 2.0,
    "radius_le": 0.02,
    "recamber_le": -2.0,
    "recamber_te": 0.5,
    "lean": 0.02,
    "thickness": 0.05,
    "xtmax": 0.1,
}


class BaseTurbineStageComp(om.ExternalCodeComp):
    """Run turbine simulation from a parameter set.

    To make a subclass:
        * Define abstract methods params_from_inputs and outputs_from_metadata
        * Declare inputs and outputs in setup method, then call super().setup()

    """

    INPUT_FILE_NAME = "params.json"
    OUTPUT_FILE_NAME = "meta.json"

    def initialize(self):
        self.options.declare("datum_params")
        self.options.declare("base_dir")
        self.options.declare("input_reference")
        self.options.declare("input_sensitivity")
        self.options.declare("runner")
        self.options["fail_hard"] = False
        self.fevals = 0

    def _normalise_var(self, k, val):
        input_reference = self.options["input_reference"]
        input_sensitivity = self.options["input_sensitivity"]
        return (val - input_reference[k]) / input_sensitivity[k] + 1.0

    def _denormalise_var(self, k, val):
        input_reference = self.options["input_reference"]
        input_sensitivity = self.options["input_sensitivity"]
        return (val - 1.0) * input_sensitivity[k] + input_reference[k]

    def _normalise_inputs(self, inputs):
        """Scale input variables to 1 at datum with prescribed sensitivity."""
        return {k + "_norm": self._normalise_var(k, inputs[k]) for k in inputs}

    def _denormalise_inputs(self, inputs_norm):
        """Unscale input variables to 1 at datum with prescribed sensitivity."""
        inputs = {}
        for kn in inputs_norm:
            k = kn.replace("_norm", "")
            inputs[k] = self._denormalise_var(k, inputs_norm[kn])
        return inputs

    def _normalise_bounds(self):
        upper_norm = self._normalise_inputs(INPUT_UPPER)
        lower_norm = self._normalise_inputs(INPUT_LOWER)
        return {k: (lower_norm[k], upper_norm[k]) for k in upper_norm}

    def _print_step_sizes(self, rhobeg):
        for k in self.options["active_design_vars"]:
            step = self._denormalise_var(k, rhobeg) - self._denormalise_var(
                k, 0.0
            )
            print("  %s: %.3f" % (k, step))

    def setup(self):
        pass

    def setup_partials(self):
        self.declare_partials(
            of="*", wrt="*", method="fd", step=0.01, step_calc="abs"
        )

    def params_from_inputs(self):
        raise NotImplementedError("Should define this method in subclasses.")

    def outputs_from_metadata(self):
        raise NotImplementedError("Should define this method in subclasses.")

    def compute(self, inputs, outputs):

        # Make a new workdir
        workdir = util.make_rundir(self.options["base_dir"])
        input_file_path = os.path.join(workdir, self.INPUT_FILE_NAME)
        output_file_path = os.path.join(workdir, self.OUTPUT_FILE_NAME)

        # Set external command options
        self.options["external_input_files"] = [input_file_path]
        self.options["external_output_files"] = [output_file_path]
        self.options["command"] = [
            self.options["runner"],
            input_file_path,
        ]

        # Use abstract method to generate the parameter set from general inputs
        param_now = self.params_from_inputs(inputs)

        # Save parameters file for TS
        param_now.to_json(input_file_path)

        try:

            super().compute(inputs, outputs)

        except om.AnalysisError:

            raise

        self.fevals += 1

        # parse the output file from the external code
        with open(output_file_path) as f:
            m = json.load(f)

        # Insert outputs in-place using abstract method
        self.outputs_from_metadata(m, outputs)


class ThreeDimensionalTurbineStageComp(BaseTurbineStageComp):
    """Run turbine simulation with varying sections up span."""

    def initialize(self):
        self.options.declare("rtol")
        self.options.declare("active_design_vars")
        super().initialize()

    def setup(self):

        param_datum = self.options["datum_params"]

        nsect = param_datum.row_sections[0].nsect

        stagger_start = np.empty((2, nsect))
        beta_start = np.empty((2, nsect))
        Rle_start = np.empty((2, nsect))
        recamber_start = np.empty((2, nsect, 2))
        thickness_start = np.empty((2, nsect, 2))
        xtmax_start = np.empty((2, nsect, 2))
        for irow in range(2):
            row_sections = param_datum.row_sections[irow]

            for j in range(nsect):

                spf_j = row_sections.spf[j]

                if row_sections.stagger[j] == 0.0:
                    chi = param_datum.free_vortex_chi(spf_j)[irow].reshape(-1)
                    stagger_start[irow, j] = np.degrees(
                        np.arctan(np.mean(np.tan(np.radians(chi))))
                    )
                else:
                    stagger_start[irow, j] = row_sections.stagger[j]

                Rle_start[irow, j] = row_sections.Rle[j]
                beta_start[irow, j] = row_sections.beta[j]

                xtmax_start[irow, j, :] = np.array(
                    (
                        row_sections.max_thickness_location_ps[j],
                        row_sections.max_thickness_location_ss[j],
                    )
                )

                recamber_start[irow, j, :] = row_sections.recamber[j]

                thickness_start[irow, j, :] = np.array(
                    (
                        row_sections.thickness_ps[j],
                        row_sections.thickness_ss[j],
                    )
                )

        # Normalise dimensional inputs to non-dimensional
        stagger_start_norm = self._normalise_var("stagger", stagger_start)
        Rle_start_norm = self._normalise_var("radius_le", Rle_start)
        beta_start_norm = self._normalise_var("beta", beta_start)
        thickness_start_norm = self._normalise_var("thickness", thickness_start)
        xtmax_start_norm = self._normalise_var("xtmax", xtmax_start)

        recamber_le_start_norm = self._normalise_var(
            "recamber_le", recamber_start[:, :, 0]
        )
        recamber_te_start_norm = self._normalise_var(
            "recamber_te", recamber_start[:, :, 1]
        )

        self.add_input("stagger_norm", val=stagger_start_norm.flatten())
        self.add_input("recamber_le_norm", val=recamber_le_start_norm.flatten())
        self.add_input("recamber_te_norm", val=recamber_te_start_norm.flatten())

        self.add_input("beta_norm", val=beta_start_norm.flatten())
        self.add_input("radius_le_norm", val=Rle_start_norm.flatten())
        self.add_input("thickness_norm", val=thickness_start_norm.flatten())
        self.add_input("xtmax_norm", val=xtmax_start_norm.flatten())

        self.add_output("lost_efficiency_percent")
        self.add_output("err_phi_rel")
        self.add_output("err_psi_rel")
        self.add_output("err_Lam_rel")
        self.add_output("efficiency")
        self.add_output("loss_split")
        self.add_output("runid")

        if nsect == 3:
            lean_start = np.copy(param_datum.row_sections[0].lean)[1]
        elif nsect == 1:
            lean_start = np.copy(param_datum.row_sections[0].lean)[0]

        lean_start_norm = self._normalise_var("lean", lean_start)
        self.add_input("lean_norm", val=lean_start_norm.flatten())

        super().setup()

    def params_from_inputs(self, inputs_norm):

        # Convert normalised optimiser variables to dimensional variables
        inputs = self._denormalise_inputs(inputs_norm)
        print("In :")
        for k in self.options["active_design_vars"]:
            print("  " + k + ": " + np.array_str(inputs[k], precision=4))

        # Make a copy of the datum parameter set
        param_datum = self.options["datum_params"]
        param_now = param_datum.copy()
        nsect = param_now.row_sections[0].nsect

        # Reshape vector inputs to arrays of correct shape
        stagger = np.reshape(inputs["stagger"], (2, nsect))
        thickness = np.reshape(inputs["thickness"], (2, nsect, 2))
        beta = np.reshape(inputs["beta"], (2, nsect))
        Rle = np.reshape(inputs["radius_le"], (2, nsect))
        xtmax = np.reshape(inputs["xtmax"], (2, nsect, 2))

        recamber_le = np.reshape(inputs["recamber_le"], (2, nsect))
        recamber_te = np.reshape(inputs["recamber_te"], (2, nsect))
        recamber = np.stack((recamber_le, recamber_te), axis=-1)

        for irow in range(2):

            row_sections = param_now.row_sections[irow]

            row_sections.stagger[:] = stagger[irow]
            row_sections.beta[:] = beta[irow]
            row_sections.Rle[:] = Rle[irow]
            row_sections.recamber[:] = recamber[irow]
            row_sections.thickness_ps[:] = thickness[irow, :, 0].T
            row_sections.thickness_ss[:] = thickness[irow, :, 1].T
            row_sections.max_thickness_location_ps[:] = xtmax[irow, :, 0].T
            row_sections.max_thickness_location_ss[:] = xtmax[irow, :, 1].T

        if "lean" in inputs:
            lean = inputs["lean"]
            if nsect == 3:
                param_now.row_sections[0].lean[1] = lean
            elif nsect == 1:
                param_now.row_sections[0].lean[0] = lean

        return param_now

    def outputs_from_metadata(self, metadata, outputs):

        params = self.options["datum_params"]
        rtol = self.options["rtol"]

        # outputs["lost_efficiency_percent"] = metadata["eta_lost"] * 100.0
        outputs["lost_efficiency_percent"] = metadata["eta_lost_wp"] * 100.0

        for v in ["phi", "psi", "Lam"]:
            outputs["err_" + v + "_rel"] = (
                metadata[v] / getattr(params, v) - 1.0
            ) / rtol

        outputs["runid"] = metadata["runid"]

        print(
            "Out %d(%012d): lost_eta = %.2f%%, err phi,psi,Lam = %+.1f,%+.1f,%+.1f%%"
            % (
                self.fevals,
                int(outputs["runid"]),
                outputs["lost_efficiency_percent"],
                outputs["err_phi_rel"] * rtol * 100.0,
                outputs["err_psi_rel"] * rtol * 100.0,
                outputs["err_Lam_rel"] * rtol * 100.0,
            )
        )

        outputs["loss_split"] = metadata["loss_rat"]
        outputs["efficiency"] = metadata["eta"]

    def update_guess(self):
        self.options["datum_params"].guess_file = self._get_file_in_run_dir(
            "output_avg.hdf5"
        )

    def update_effy_loss_split(self):
        """One-time update of effy and loss split."""
        params = self.options["datum_params"]
        # We set eta and loss to copies of most recent values
        params.eta_guess = self.get_val("efficiency") + 0.0
        params.loss_split = self.get_val("loss_split") + 0.0

    def track_effy_loss_split(self):
        """Enable tracking of effy and loss split during optimisation."""
        params = self.options["datum_params"]
        # We set eta and loss to references that point to most recent values
        params.eta_guess = self.get_val("efficiency")
        params.loss_split = self.get_val("loss_split")

    def freeze_effy_loss_split(self):
        """Disable tracking of effy and loss split during optimisation."""
        # We set eta and loss to a copy of their current values, breaking ref
        params = self.options["datum_params"]
        params.eta_guess = params.eta_guess + 0.0
        params.loss_split = params.loss_split + 0.0

    def _get_file_in_run_dir(self, f):
        return os.path.join(
            self.options["base_dir"], "%012d" % self.get_val("runid"), f
        )

    def get_metadata(self):
        meta_path = self._get_file_in_run_dir(self.OUTPUT_FILE_NAME)
        with open(meta_path) as f:
            meta = json.load(f)
        return meta

    def get_params(self):
        params_path = self._get_file_in_run_dir(self.INPUT_FILE_NAME)
        return three_dimensional_stage.StageParameterSet.from_json(params_path)

    def get_constraint_errors(self):
        err_rel = np.array(
            [
                self.get_val("err_phi_rel"),
                self.get_val("err_psi_rel"),
                self.get_val("err_Lam_rel"),
            ]
        )
        return err_rel

    def is_on_target(self, thresh_fac=1.0):
        return np.all(np.abs(self.get_constraint_errors()) < thresh_fac)


def run_once(params, runner):
    base_dir = os.path.abspath("./run_once")

    # Set up model
    prob = om.Problem()
    model = prob.model
    model.add_subsystem(
        "ts",
        ThreeDimensionalTurbineStageComp(
            datum_params=params,
            base_dir=base_dir,
            rtol=1.0,
            input_reference=INPUT_REFERENCE,
            input_sensitivity=INPUT_SENSITIVITY,
            runner=runner,
        ),
        promotes=["*"],
    )

    # Setup problem
    prob.setup()

    print("** Running once")
    prob.run_model()


def _initialise_turbine(params, runner, base_dir, tol, active_design_vars):
    """Make an OpenMDAO problem with turbine model, specify active vars."""

    log_path = os.path.join(base_dir, LOG_SQL)

    print("Base dir:\n %s" % base_dir)
    print("Logging to:\n %s" % log_path)

    # Set up model
    prob = om.Problem()
    model = prob.model
    model.add_subsystem(
        "turbine",
        ThreeDimensionalTurbineStageComp(
            datum_params=params,
            base_dir=base_dir,
            rtol=tol,
            input_reference=INPUT_REFERENCE,
            input_sensitivity=INPUT_SENSITIVITY,
            runner=runner,
            active_design_vars=active_design_vars,
        ),
        promotes=["*"],
    )

    bounds_norm = model.turbine._normalise_bounds()
    print("Bounds:")
    for k in active_design_vars:
        kn = k + "_norm"
        print(
            "  "
            + k
            + ": "
            + str(
                [model.turbine._denormalise_var(k, v) for v in bounds_norm[kn]]
            )
        )
        prob.model.add_design_var(kn, *bounds_norm[kn])

    # Constraints
    prob.model.add_constraint("err_phi_rel", lower=-1.0, upper=1.0)
    prob.model.add_constraint("err_psi_rel", lower=-1.0, upper=1.0)
    prob.model.add_constraint("err_Lam_rel", lower=-1.0, upper=1.0)

    # Objective
    prob.model.add_objective("lost_efficiency_percent")

    # Set up optimizer
    driver = prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "COBYLA"
    prob.driver.opt_settings["maxiter"] = 100
    prob.driver.opt_settings["rhobeg"] = 1.0
    prob.driver.opt_settings["tol"] = 0.4 #changed this from 0.25 to reduce eta convergence
    print("Initial step sizes:")
    model.turbine._print_step_sizes(prob.driver.opt_settings["rhobeg"])

    # Set up recorder
    driver.recording_options["includes"] = ["*"]
    driver.recording_options["record_objectives"] = True
    driver.recording_options["record_constraints"] = True
    driver.recording_options["record_desvars"] = True
    driver.recording_options["record_inputs"] = True
    driver.recording_options["record_outputs"] = True
    driver.recording_options["record_residuals"] = True
    recorder = om.SqliteRecorder(log_path)
    driver.add_recorder(recorder)

    # Setup problem
    prob.setup()

    return prob, prob.model.turbine


def optimise_midspan(params, runner, stem="./", tol=0.02):
    """Optimise turbine geometry with midspan design variables.

    Thickness, leading edge radius and wedge angle are constant up span.
    Recamber is constant up span, but metal angles vary like a free vortex.
    Stagger relative to a datum stagger is constant up the span.

    Starts with a crude initial guess, gradually ramping Mach if needed
    for robustness.
    """

    base_dir = os.path.abspath(stem)

    active_design_vars = [
        "recamber_te",
        "stagger",
        "xtmax",
        "thickness",
        "radius_le",
        "beta",
        "lean",
    ]

    prob, turbine = _initialise_turbine(
        params, runner, base_dir, tol, active_design_vars
    )

    # Mixing length initial guess
    params.guess_file = None
    params.cfd_config["ilos"] = 1
    params.cfd_config["dampin"] = 3.0
    Ma_lim = 0.65
    if params.Ma2 > Ma_lim:
        Ma_target = params.Ma2
        nMa = 1 + np.ceil((Ma_target - Ma_lim) / 0.1).astype(int)
        for Ma in np.linspace(Ma_lim, Ma_target, nMa):
            print("** Running ML guess, Ma=%.2f" % Ma)
            params.Ma2 = Ma
            prob.run_model()
            turbine.update_guess()

        print("** Running SA guess, high damping")
        params.cfd_config["ilos"] = 2
        prob.run_model()
        turbine.update_guess()

    else:

        print("** Running ML guess")
        prob.run_model()
        turbine.update_guess()
        turbine.update_effy_loss_split()

    print("** Running SA guess")
    params.cfd_config["ilos"] = 2
    params.cfd_config["dampin"] = 25.0
    params.cfd_config["nchange"] = 0
    prob.run_model()
    turbine.update_guess()
    turbine.update_effy_loss_split()

    # Enable tracking of efficiency and loss split
    # Should hopefully not confuse optimiser if step size is small enough
    # turbine.track_effy_loss_split()

    # Enable tracking of efficiency and loss split
    # Should hopefully not confuse optimiser if step size is small enough

    print("** Correcting effy and loss split")
    prob.run_model()
    turbine.update_guess()
    turbine.update_effy_loss_split()

    print("** Fixing annulus line")
    prob.run_model()
    turbine.update_guess()

    # From now on, annulus line is frozen
    # Don't need new meshes each time
    params.grid_type = "warp"

    # Now do optimisation
    # print("** Running optimisation")
    # prob.run_driver()                 #COMMENT THIS OUT TO SKIP OPTIMISATION

    # # If far away from target, tweak exit angles
    # if not turbine.is_on_target(thresh_fac=2.0):
    #     print("** Restarting optimisation")
    #     # Make a new problem with tighter tolerance
    #     prob, turbine = _initialise_turbine(
    #         turbine.get_params(),
    #         runner,
    #         base_dir,
    #         tol * 0.5,
    #         [
    #             "recamber_te",
    #         ],
    #     )
    #     # If we just restart, then the old convergence history will be
    #     # overwritten. So move the old case recording file.
    #     shutil.copy(
    #         os.path.join(base_dir, LOG_SQL),
    #         os.path.join(base_dir, LOG_SQL.replace(".sql", "_old.sql")),
    #     )
    #     print("** Tweaking exit angles")
    #     prob.run_driver()

    # Return optimal stuff
    params_out = turbine.get_params()
    meta_out = turbine.get_metadata()

    return params_out, meta_out


def optimise_radial(params, runner, stem, tol):
    """Optimise radial variations in turbine geometry."""

    base_dir = os.path.abspath(stem)

    active_design_vars = [
        "recamber_te",
        "lean",
        "recamber_le",
        "stagger",
    ]
    prob, turbine = _initialise_turbine(
        params, runner, base_dir, tol, active_design_vars
    )

    print("** Running initial point")
    prob.run_model()
    turbine.update_guess()
    turbine.update_effy_loss_split()

    # Now do optimisation
    print("** Running optimisation")
    prob.run_driver()

    # Return optimal stuff
    params_out = turbine.get_params()
    meta_out = turbine.get_metadata()

    return params_out, meta_out
