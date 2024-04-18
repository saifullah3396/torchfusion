from torchvision.transforms import Compose

from torchfusion.core.data.factory.data_augmentation import DataAugmentationFactory
from torchfusion.core.data.utilities.containers import TransformsDict


def load_transforms_from_config(train_augs, eval_augs):
    # define data transforms according to the configuration
    tf = TransformsDict()
    if train_augs is not None:
        tf.train = []
        for aug_args in train_augs:
            aug = DataAugmentationFactory.create(
                aug_args.name,
                aug_args.kwargs,
            )
            tf.train.append(aug)

    if eval_augs is not None:
        tf.validation = []
        tf.test = []
        for aug_args in eval_augs:
            aug = DataAugmentationFactory.create(
                aug_args.name,
                aug_args.kwargs,
            )
            tf.validation.append(aug)
            tf.test.append(aug)

    # wrap the transforms in a callable class
    if tf.train is not None:
        tf.train = Compose(tf.train)

    if tf.validation is not None:
        tf.validation = Compose(tf.validation)

    if tf.test is not None:
        tf.test = Compose(tf.test)

    return tf
