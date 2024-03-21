srun --gpus-per-task=0 \
  --container-image=/netscratch/$USER/envs/xai_torch_v5.sqsh \
  --container-workdir=/home/$USER/projects/torchfusion \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,/home/$USER/:/home/$USER/ \
  --container-save=/netscratch/$USER/envs/xai_torch_v6.sqsh \
  --mem=60GB \
  --partition V100-16GB \
  --pty /bin/bash
