"""Functions for running an optimisation as queue job."""
import os, shutil, subprocess, json
from . import optimise, three_dimensional_stage, plot_optimiser


TURBIGEN_ROOT = "/".join(__file__.split("/")[:-2])

OPT_SLURM_TEMPLATE = os.path.join(TURBIGEN_ROOT, "turbigen", "submit.sh")

DATUM_NAME = "datum.json"


def run_search(param, runner, opt_name, dependent=None, begin_time=None):
    """Given a parameter set, run optimisation."""

    if not "SSH_AUTH_SOCK" in os.environ:
        raise Exception(
            "Need ssh-agent running first. Run:\neval `ssh-agent` && ssh-add"
        )

    # # Check for supersonic rotor exit flow
    # Ma_rel_rotor_exit = param.nondimensional_stage.Marel[-1]
    # if Ma_rel_rotor_exit > 1.0:
    #     raise Exception(
    #         "Aborting, supersonic rotor exit Ma = %.2f" % Ma_rel_rotor_exit
    #     )

    # Set up directory to hold all results for this optimisation
    base_dir = os.path.join(TURBIGEN_ROOT, "run", opt_name)
    if not os.path.isdir(base_dir):
        os.mkdir(base_dir)

    # print("Starting optimisation in base dir: %s" % base_dir)

    os.chdir(base_dir)

    # Write out datum parameters
    datum_file = os.path.join(base_dir, DATUM_NAME)
    param.to_json(datum_file)

    # Copy in SLURM script
    slurm_file = os.path.join(base_dir, "submit.sh")
    shutil.copy(OPT_SLURM_TEMPLATE, slurm_file)

    # Append optimisation name to the slurm job name
    sedcmd = "sed -i 's?jobname?turbigen_%s?' %s" % (opt_name, slurm_file)
    os.system(sedcmd)

    # Fill in working directory
    sedcmd = "sed -i 's?workdir?%s?g' %s" % (base_dir, slurm_file)
    os.system(sedcmd)

    # Fill in runner
    sedcmd = "sed -i 's?runner?%s?g' %s" % (os.path.abspath(runner), slurm_file)
    os.system(sedcmd)

    # Build up the sbatch command
    cmd_str = "sbatch "
    if dependent:
        cmd_str += "-d afterany:%d " % dependent
    if begin_time:
        cmd_str += "-b %s " % begin_time
    cmd_str += slurm_file

    # Run sbatch
    jid = _parse_sbatch(
        subprocess.check_output(
            cmd_str,
            cwd=base_dir,
            shell=True,
        )
    )

    return jid


def _parse_sbatch(s):
    return int(s.decode("utf-8").strip().split(" ")[-1])


def _run_search(base_dir, runner):
    """SLURM script calls this in the optimisation dir."""

    # Set up some file paths
    datum_file = os.path.join(base_dir, DATUM_NAME)
    if not os.path.isfile(datum_file):
        raise Exception("No datum parameters found.")

    # Read datum parameters
    param = three_dimensional_stage.StageParameterSet.from_json(datum_file)

    print("Loaded datum parameters:\n %s" % datum_file)

    # Run the optimisation
    param_opt, meta_opt = optimise.optimise_midspan(
        param, runner, stem=base_dir, tol=0.02
    )

    # Write out optimum params
    opt_file = os.path.join(base_dir, "optimised_params.json")
    param_opt.to_json(opt_file)

    # Write out optimum metadata
    meta_file = os.path.join(base_dir, "optimised_meta.json")
    with open(meta_file, "w") as f:
        json.dump(meta_opt, f, indent=4)

    # Make plots

    try:
        op = plot_optimiser.OptimisePlotter(
            os.path.join(base_dir, "opt_old.sql"), "_old"
        )
        op.convergence()
        op.constraints()
        op.steps()
    except:
        pass

    try:
        op = plot_optimiser.OptimisePlotter(os.path.join(base_dir, "opt.sql"))
        op.convergence()
        op.constraints()
        op.steps()
    except:
        pass
