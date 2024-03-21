#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# make sure only first task per node installs stuff, others wait
DONEFILE="/tmp/install_done_${SLURM_JOBID}"
if [[ $SLURM_LOCALID == 0 ]]; then
  # pip install fire

  # # install python dependencies
  # pip install -r $SCRIPT_DIR/../../cluster_requirements.txt

  # # install libraries
  # apt update
  # apt install libgl1-mesa-glx -y
  # source /netscratch/saifullah/envs/TORCH_FUSION/bin/activate/
  pip install -U datasets==2.13.0
  pip install -U diffusers
  pip install fsspec==2023.9.2

  # Tell other tasks we are done installing
  touch "${DONEFILE}"
else
  # Wait until packages are installed
  while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
fi