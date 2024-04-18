#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# # with augmentation/transforms applied to dataset during data loading. Default is BasicImageAug
# $SCRIPT_DIR/../../../../scripts/train.sh \
#     +train=image_classification/with_aug \
#     args/data_args=image_classification/cifar10 \
#     args/model_args=fusion_model_toy_model_cifar10 \
#     image_size_y=32 \
#     image_size_x=32 \
#     per_device_train_batch_size=64 \
#     per_device_eval_batch_size=64 \
#     dataloader_num_workers=4 \
#     experiment_name=train_cifar10 \
#     max_epochs=10 \
#     test_run=True \
#     num_labels=10 \
#     $@

# with augmentation/transforms applied to dataset during data loading. Default is BasicImageAug
# model used tv_model_alexnet
# $SCRIPT_DIR/../../../../scripts/train.sh \
#     +train=image_classification/with_aug \
#     args/data_args=image_classification/cifar10 \
#     args/model_args=tv_model_alexnet \
#     image_size_y=224 \
#     image_size_x=224 \
#     per_device_train_batch_size=64 \
#     per_device_eval_batch_size=64 \
#     dataloader_num_workers=4 \
#     experiment_name=train_cifar10 \
#     max_epochs=10 \
#     test_run=False \
#     num_labels=10 \
#     $@

# with augmentation/transforms applied to dataset during data loading. Default is BasicImageAug
# model used timm_model_resnet50
# $SCRIPT_DIR/../../../../scripts/train.sh \
#     +train=image_classification/with_aug \
#     args/data_args=image_classification/cifar10 \
#     args/model_args=timm_model_resnet50 \
#     image_size_y=224 \
#     image_size_x=224 \
#     per_device_train_batch_size=64 \
#     per_device_eval_batch_size=64 \
#     dataloader_num_workers=4 \
#     experiment_name=train_cifar10 \
#     max_epochs=10 \
#     test_run=False \
#     num_labels=10 \
#     $@

# with augmentation/transforms applied to dataset during data loading. Default is BasicImageAug
# model used transformer_model based vit: google/vit-base-patch16-224
$SCRIPT_DIR/../../../../scripts/train.sh \
    +train=image_classification/with_aug \
    args/data_args=image_classification/cifar10 \
    args/model_args=transformer_model/vit-base-patch16-224 \
    image_size_y=224 \
    image_size_x=224 \
    per_device_train_batch_size=64 \
    per_device_eval_batch_size=64 \
    dataloader_num_workers=4 \
    experiment_name=train_cifar10 \
    max_epochs=10 \
    test_run=False \
    num_labels=10 \
    $@
