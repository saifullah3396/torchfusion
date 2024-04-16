#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export PYTHONPATH=$SCRIPT_DIR/../src:$SCRIPT_DIR/../external/torchfusion/src

# pretrain the compvis - klf4 encoder from compvis/latent-diffusion repository
$SCRIPT_DIR/../../../../scripts/train.sh -c vae_pretraining \
    +experiments=pretrain_kl_aegan \
    dataset_config_name=default \
    args/data_args=tobacco3482 \
    args/model_args=compvis_autoencoder/f4 \
    checkpoint=/home/ataraxia/Projects/models/compvis/klf4.pt \
    $@

# pretrain the diffusers/stable-diffusion-v1-4 autoencoder from pretrained weights
# $SCRIPT_DIR/../../../../scripts/train.sh -c vae_pretraining \
#     +experiments=pretrain_kl_aegan \
#     dataset_config_name=default \
#     args/data_args=tobacco3482 \
#     args/model_args=diffusers/stable-diffusion-v1-4 \
#     $@

# # pretrain the diffusers/stabilityai-sdxl-vae autoencoder from pretrained weights
# $SCRIPT_DIR/../../../../scripts/train.sh -c vae_pretraining \
#     +experiments=pretrain_kl_aegan \
#     dataset_config_name=default \
#     args/data_args=tobacco3482 \
#     args/model_args=diffusers/stabilityai-sdxl-vae \
#     $@
