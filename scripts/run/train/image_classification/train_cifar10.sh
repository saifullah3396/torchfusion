#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# # with augmentation/transforms applied to dataset during data loading. Default is BasicImageAug
$SCRIPT_DIR/../../../../scripts/train.sh \
    +train=image_classification/with_aug \
    args/data_args=image_classification/cifar10 \
    args/model_args=fusion_model_toy_model_cifar10 \
    image_size_y=32 \
    image_size_x=32 \
    per_device_train_batch_size=64 \
    per_device_eval_batch_size=64 \
    dataloader_num_workers=4 \
    experiment_name=train_cifar10 \
    max_epochs=10 \
    test_run=True \
    num_labels=10 \
    $@

# args/train_val_sampler=random_split \
# with augmentation/transforms applied to dataset during data loading and preprocessing. Default data loading transform is BasicImageAug and default preprocessing transform is ImagePreprocess
# $SCRIPT_DIR/../../../../scripts/train.sh \
#     +train=image/with_aug_and_preprocess \
#     args/data_args=image_classification/cifar10 \
#     image_size_y=224 \
#     image_size_x=224 \
#     preprocess_image_size_x=224 \
#     preprocess_image_size_y=224 # use preprocessing augmentation instead to convert all images to tensor

# ./scripts/train.sh -c image_classification +run=train_model args/data_args=cifar10 args/model_args=fusion_model_toy_model_cifar10 realtime_image_size_x=32 realtime_image_size_y=32
# ./scripts/train.sh -c image_classification +run=train_model args/data_args=cifar10 args/model_args=tv_model_alexnet realtime_image_size_x=224 realtime_image_size_y=224
# ./scripts/train.sh -c image_classification +run=train_model args/data_args=cifar10 args/model_args=timm_model_resnet50 realtime_image_size_x=224 realtime_image_size_y=224
# ./scripts/train.sh -c image_classification +run=train_model args/data_args=cifar10 args/model_args=transformers_model realtime_image_size_x=224 realtime_image_size_y=224
