#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
$SCRIPT_DIR/../../../../scripts/analyze.sh \
    -c prepare_datasets \
    +run=prepare_image_dataset \
    dataset_config_name=default \
    args/data_args=image_classification/tobacco3482 \
    visualize=true \
    realtime_image_size_x=512 \
    realtime_image_size_y=512
