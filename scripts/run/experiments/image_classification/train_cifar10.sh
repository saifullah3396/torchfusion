./scripts/train.sh -c image_classification +run=train_model args/data_args=cifar10 args/model_args=fusion_model_toy_model_cifar10 realtime_image_size_x=32 realtime_image_size_y=32
./scripts/train.sh -c image_classification +run=train_model args/data_args=cifar10 args/model_args=tv_model_alexnet realtime_image_size_x=224 realtime_image_size_y=224
./scripts/train.sh -c image_classification +run=train_model args/data_args=cifar10 args/model_args=timm_model_resnet50 realtime_image_size_x=224 realtime_image_size_y=224
./scripts/train.sh -c image_classification +run=train_model args/data_args=cifar10 args/model_args=transformers_model
