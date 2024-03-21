from dataclasses import dataclass

from torchvision import transforms

from torchfusion.core.data.data_augmentations.base import DataAugmentation


@dataclass
class Cifar10ToyAug(DataAugmentation):
    def __str__(self):
        return str(self._aug)

    def _initialize_aug(self):
        aug = [
            transforms.Resize(
                (32, 32)
            ),  # resises the image so it can be perfect for our model.
            transforms.RandomHorizontalFlip(),  # FLips the image w.r.t horizontal axis
            transforms.RandomRotation(10),  # Rotates the image to a specified angel
            transforms.RandomAffine(
                0, shear=10, scale=(0.8, 1.2)
            ),  # Performs actions like zooms, change shear angles.
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2
            ),  # Set the color params
            transforms.ToTensor(),  # comvert the image to tensor so that it can work with torch
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            ),  # Normalize all the images
        ]

        # generate torch transformation
        return transforms.Compose(aug)

    def __post_init__(self):
        self._aug = self._initialize_aug()

    def __call__(self, sample):
        if isinstance(sample, list):
            return [self._aug(s) for s in sample]
        else:
            return self._aug(sample)
