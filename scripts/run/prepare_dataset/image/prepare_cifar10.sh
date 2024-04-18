#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# # with no augmentation/transforms applied to dataset during data loading
# $SCRIPT_DIR/../../../../scripts/prepare_dataset.sh \
#     +prepare_dataset=image/no_aug \
#     args/data_args=image_classification/cifar10 # this will fail due to batch collation

# # with augmentation/transforms applied to dataset during data loading. Default is BasicImageAug
$SCRIPT_DIR/../../../../scripts/prepare_dataset.sh \
    +prepare_dataset=image/with_aug \
    args/data_args=image_classification/cifar10 \
    image_size_y=224 \
    image_size_x=224 # use augmentation instead to convert all images to tensor

# with augmentation/transforms applied to dataset during data loading and preprocessing. Default data loading transform is BasicImageAug and default preprocessing transform is ImagePreprocess
# $SCRIPT_DIR/../../../../scripts/prepare_dataset.sh \
#     +prepare_dataset=image/with_aug_and_preprocess \
#     args/data_args=image_classification/cifar10 \
#     image_size_y=224 \
#     image_size_x=224 \
#     preprocess_image_size_x=224 \
#     preprocess_image_size_y=224 # use preprocessing augmentation instead to convert all images to tensor
