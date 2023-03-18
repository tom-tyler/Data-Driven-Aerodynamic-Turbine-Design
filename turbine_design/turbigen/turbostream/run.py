"""From a parameters JSON file, start Turbostream on a free GPU, and then
post-process the flow solution to get an output metadata JSON."""

import sys, os, subprocess, time
from turbigen import three_dimensional_stage
try:
    from . import write, post_process
except (ValueError,ImportError):
    from turbigen.turbostream import write, post_process

if __name__ == "__main__":

    # Get argument
    try:
        json_file_path = sys.argv[1]
    except IndexError:
        raise Exception("No input file specified.")
        sys.exit(1)

    # Work out some file names
    workdir, json_file_name = os.path.split(os.path.abspath(json_file_path))
    basedir = os.path.dirname(workdir)

    gpu_id = 0

    # Change to working dir
    os.chdir(os.path.abspath(workdir))

    start = time.time()

    # Read the parameters
    param = three_dimensional_stage.StageParameterSet.from_json(json_file_name)

    # Write the grid
    input_file_name = "input.hdf5"
    write.write_stage_from_params(param, input_file_name)
    mesh_elapsed = time.time() - start
    print("Written input hdf5 in %d s" % mesh_elapsed)

    start = time.time()

    output_prefix = "output"
    # Start Turbostream
    cmd_str = "CUDA_VISIBLE_DEVICES=%d turbostream %s %s 1 > log.txt" % (
        gpu_id,
        input_file_name,
        output_prefix,
    )
    subprocess.Popen(cmd_str, shell=True).wait()

    run_elapsed = time.time() - start
    print("Ran CFD in %d s" % run_elapsed)

    start = time.time()

    # Post process
    post_process.post_process(output_prefix + "_avg.hdf5")

    # Remove extraneous files
    spare_files = [ "stopit",
        output_prefix + ".xdmf",
        output_prefix + "_avg.xdmf",
        output_prefix + ".hdf5",
        "input.hdf5",
    ]
    for f in spare_files:
        try:
            os.remove(os.path.join(".", f))
        except OSError:
            pass

    post_elapsed = time.time() - start
    print("Post-processed in %d s" % post_elapsed)
