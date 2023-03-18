#!/bin/bash
#
# A slim wrapper around run_turbostream.py that CLEARS existing environment and
# sources a version of the official env without MPI, but saves SSH agent
TS_ENV='/usr/local/software/turbostream/ts3610_a100/bashrc_module_ts3610_a100'
source $TS_ENV
python -m turbigen.turbostream.run $@
# env -i bash -c "PYTHONPATH=$PYTHONPATH source $TS_ENV && echo $TSHOME && SSH_AUTH_SOCK=$SSH_AUTH_SOCK SSH_AGENT_PID=$SSH_AGENT_PID python -m turbigen.turbostream.run $@"
