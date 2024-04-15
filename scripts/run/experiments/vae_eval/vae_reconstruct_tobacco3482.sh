#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export PYTHONPATH=$SCRIPT_DIR/../src:$SCRIPT_DIR/../external/torchfusion/src

# evaluate the compvis - klf4 encoder from compvis/latent-diffusion repository
# $SCRIPT_DIR/../../../../scripts/analyze.sh -c vae_eval \
#     +analysis=vae_reconstruct_with_preprocess \
#     dataset_config_name=default \
#     args/data_args=tobacco3482 \
#     realtime_image_size_x=256 \
#     realtime_image_size_y=256 \
#     args/model_args=compvis_autoencoder/f4 \
#     checkpoint=/home/ataraxia/Projects/models/compvis/klf4.pt \
#     with_amp_inference=False $@

# evaluate the compvis - klf4 autoencoder from compvis/latent-diffusion repository finetuned on IIT-CDIP dataset with our own implementation
# this implementation includes augmentations etc
# this results in FID of 0.9927929809670957 on Tobacco3482 train dataset
# $SCRIPT_DIR/../../../../scripts/analyze.sh -c vae_eval \
#     +analysis=vae_reconstruct_with_preprocess \
#     dataset_config_name=default \
#     args/data_args=tobacco3482 \
#     realtime_image_size_x=256 \
#     realtime_image_size_y=256 \
#     args/model_args=compvis_autoencoder/f4 \
#     checkpoint=/home/ataraxia/Projects/models/torchfusion/compvis_kl_f4_IIT_CDIP.pt \
#     checkpoint_state_dict_key=model \
#     with_amp_inference=False $@

# evaluate the diffusers/stable-diffusion-v1-4 autoencoder from pretrained weights
# this results in FID of 11.78965433341861 on Tobacco3482 train dataset
# $SCRIPT_DIR/../../../../scripts/analyze.sh -c vae_eval \
#     +analysis=vae_reconstruct_with_preprocess \
#     dataset_config_name=default \
#     args/data_args=tobacco3482 \
#     realtime_image_size_x=256 \
#     realtime_image_size_y=256 \
#     args/model_args=diffusers/stable-diffusion-v1-4 \
#     with_amp_inference=False $@

# # evaluate the diffusers/stable-diffusion-v1-4 finetuned on IIT-CDIP dataset with our own implementation
# # this implementation includes augmentations etc
# $SCRIPT_DIR/../../../../scripts/analyze.sh -c vae_eval \
#     +analysis=vae_reconstruct_with_preprocess \
#     dataset_config_name=default \
#     args/data_args=tobacco3482 \
#     realtime_image_size_x=256 \
#     realtime_image_size_y=256 \
#     args/model_args=diffusers/stable-diffusion-v1-4 \
#     checkpoint=no_checkpoint.pt \
#     checkpoint_state_dict_key=model \
#     with_amp_inference=False $@

# evaluate the diffusers/stabilityai-sdxl-vae autoencoder from pretrained weights
# this results in FID of 8.064821253021336 on Tobacco3482 train dataset
$SCRIPT_DIR/../../../../scripts/analyze.sh -c vae_eval \
    +analysis=vae_reconstruct_with_preprocess \
    dataset_config_name=default \
    args/data_args=tobacco3482 \
    realtime_image_size_x=256 \
    realtime_image_size_y=256 \
    args/model_args=diffusers/stabilityai-sdxl-vae \
    with_amp_inference=False $@
