#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export PYTHONPATH=$SCRIPT_DIR/../src:$SCRIPT_DIR/../external/torchfusion/src

# $SCRIPT_DIR/../../scripts/analyze.sh -c vae_eval +analysis=vae_reconstruct_with_preprocess args/data_args=tobacco3482 dataset_config_name=with_ocr realtime_image_size_x=1024 realtime_image_size_y=1024 args/model_args=stabilityai-sdxl-vae bs=1
# $SCRIPT_DIR/../../scripts/analyze.sh -c vae_eval +analysis=vae_reconstruct_with_preprocess args/data_args=tobacco3482 dataset_config_name=with_ocr realtime_image_size_x=512 realtime_image_size_y=512 args/model_args=stabilityai-sdxl-vae bs=1
# $SCRIPT_DIR/../../scripts/analyze.sh -c vae_eval +analysis=vae_reconstruct_with_preprocess args/data_args=tobacco3482 dataset_config_name=with_ocr realtime_image_size_x=256 realtime_image_size_y=256 args/model_args=stabilityai-sdxl-vae bs=8
# $SCRIPT_DIR/../../scripts/analyze.sh -c vae_eval +analysis=vae_reconstruct_with_preprocess args/data_args=tobacco3482 dataset_config_name=with_ocr realtime_image_size_x=1024 realtime_image_size_y=1024 args/model_args=stable-diffusion-v1-4 bs=1
# $SCRIPT_DIR/../../scripts/analyze.sh -c vae_eval +analysis=vae_reconstruct_with_preprocess args/data_args=tobacco3482 dataset_config_name=with_ocr realtime_image_size_x=512 realtime_image_size_y=512 args/model_args=stable-diffusion-v1-4 bs=1
# $SCRIPT_DIR/../../scripts/analyze.sh -c vae_eval +analysis=vae_reconstruct_with_preprocess args/data_args=tobacco3482 dataset_config_name=with_ocr realtime_image_size_x=256 realtime_image_size_y=256 args/model_args=stable-diffusion-v1-4 bs=8
# $SCRIPT_DIR/../../scripts/analyze.sh -c vae_eval +analysis=vae_reconstruct_with_preprocess args/data_args=tobacco3482 dataset_config_name=with_ocr realtime_image_size_x=1024 realtime_image_size_y=1024 args/model_args=fusion_vae_kl_f4 bs=1

# $SCRIPT_DIR/../../scripts/analyze.sh -c vae_eval +analysis=vae_reconstruct_with_preprocess dataset_config_name=with_ocr args/data_args=tobacco3482 realtime_image_size_x=1024 realtime_image_size_y=1024 args/model_args=stabilityai-sdxl-vae-fp16 with_amp_inference=True bs=1
# $SCRIPT_DIR/../../scripts/analyze.sh -c vae_eval +analysis=vae_reconstruct_with_preprocess dataset_config_name=with_ocr args/data_args=tobacco3482 realtime_image_size_x=512 realtime_image_size_y=512 args/model_args=stabilityai-sdxl-vae-fp16 with_amp_inference=True
# $SCRIPT_DIR/../../../../scripts/analyze.sh -c vae_eval +analysis=vae_reconstruct_with_preprocess dataset_config_name=default args/data_args=tobacco3482 realtime_image_size_x=256 realtime_image_size_y=256 args/model_args=stabilityai-sdxl-vae-fp16 with_amp_inference=True
$SCRIPT_DIR/../../../../scripts/analyze.sh -c vae_eval +analysis=vae_reconstruct_with_preprocess dataset_config_name=default args/data_args=tobacco3482 realtime_image_size_x=256 realtime_image_size_y=256 args/model_args=compvis_autoencoder/f4 with_amp_inference=False $@
