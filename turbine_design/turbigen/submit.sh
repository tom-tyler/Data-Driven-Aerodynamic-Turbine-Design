#!/bin/bash
#SBATCH -J jobname
#SBATCH -p ampere
#SBATCH -A BRIND-SL3-GPU
#SBATCH --mail-type=FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

# export TURBIGEN_ROOT='/rds/project/gp10006/rds-gp10006-pullan-mhi/jb753/turbigen'
# export PYTHONPATH="$TURBIGEN_ROOT:$PYTHONPATH"
# export PATH="$TURBIGEN_ROOT:$PATH"
export PYTHONPATH
echo $PYTHONPATH

cd workdir
SSH_AUTH_SOCK=$SSH_AUTH_SOCK SSH_AGENT_PID=$SSH_AGENT_PID python -u -c 'from turbigen import submit; submit._run_search("workdir","runner")' &> log_opt.txt
