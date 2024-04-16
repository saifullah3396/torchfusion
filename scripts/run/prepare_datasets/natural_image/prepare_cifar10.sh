#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
$SCRIPT_DIR/../../../../scripts/analyze.sh -c prepare_datasets +run=prepare_image_dataset dataset_config_name=default args/data_args=datasets/image_classification/cifar10 visualize=true image_size_x=224 image_size_y=224
