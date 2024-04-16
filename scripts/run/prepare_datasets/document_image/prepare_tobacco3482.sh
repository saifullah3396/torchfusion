#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# without preprocess
# $SCRIPT_DIR/../../../../scripts/analyze.sh \
#     -c prepare_datasets \
#     +run=prepare_image_dataset \
#     dataset_config_name=default \
#     args/data_args=datasets/image_classification/tobacco3482 \
#     visualize=true \
#     image_size_x=512 \
#     image_size_y=512 \
#     $@

# with preprocess
$SCRIPT_DIR/../../../../scripts/analyze.sh \
    -c prepare_datasets \
    +run=prepare_image_dataset_with_preprocess \
    dataset_config_name=default \
    args/data_args=datasets/image_classification/tobacco3482 \
    visualize=true \
    preprocess_image_size_x=1024 \
    preprocess_image_size_y=1024 \
    image_size_x=512 \
    image_size_y=512 \
    $@
