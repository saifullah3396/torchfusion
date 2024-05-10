from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from io import BytesIO
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from PIL import Image as PILImage
from timm.data import create_transform
from torchfusion.core.args.args_base import ClassInitializerArgs
from torchfusion.core.constants import DataKeys
from torchfusion.core.data.data_augmentations.transforms import SquarePad
from torchfusion.core.data.factory.data_augmentation import DataAugmentationFactory
from torchfusion.core.utilities.logging import get_logger
from torchvision import transforms

from .base import DataAugmentation
from .distortions import Solarization
from .general import GrayScaleToRGB, RGBToBGR, RGBToGrayScale
from .noise import GaussianNoiseRGB

if TYPE_CHECKING:
    import torch
logger = get_logger(__name__)


def check_image_size(sample, image):
    """
    Raise an error if the image does not match the size specified in the dict.
    """
    if DataKeys.IMAGE_WIDTH in sample or DataKeys.IMAGE_HEIGHT in sample:
        # image has h, w, c as numpy array
        image_wh = (image.shape[1], image.shape[0])
        expected_wh = (
            sample[DataKeys.IMAGE_WIDTH],
            sample[DataKeys.IMAGE_HEIGHT],
        )
        if not image_wh == expected_wh:
            raise ValueError(
                "Mismatched image shape{}, got {}, expect {}.".format(
                    (
                        " for image " + sample["file_name"]
                        if "file_name" in sample
                        else ""
                    ),
                    image_wh,
                    expected_wh,
                )
                + " Please check the width/height in your annotation."
            )

    # To ensure bbox always remap to original image size
    if DataKeys.IMAGE_HEIGHT not in sample or sample[DataKeys.IMAGE_HEIGHT] is None:
        sample[DataKeys.IMAGE_HEIGHT] = image.shape[0]
    if DataKeys.IMAGE_WIDTH not in sample or sample[DataKeys.IMAGE_WIDTH] is None:
        sample[DataKeys.IMAGE_WIDTH] = image.shape[1]


def detectron2_preprocess_transform_image_and_objects(sample, geometric_tf):
    from detectron2.data.detection_utils import transform_instance_annotations
    from detectron2.data.transforms import apply_transform_gens

    # we always read image in RGB format in the dataset, when it reaches here the image is of numpay array with shape (h, w, c)
    # detectron2 needs image of shape (h, w, c) and in this place of transformation.
    # here we once resize the image to max possible sizes needed during training/testing
    image = sample[DataKeys.IMAGE]

    # sample must contain image height and image width as done for coco type datasets
    # here we assume that the sample has those. If not the image width and heights are set
    check_image_size(sample, image)

    # here the image is resized to correct aspect ratio
    # the returned geometric_tf is needed for bbox transformation in detectron2
    image, geometric_tf = apply_transform_gens(geometric_tf, image)

    # store the image shape here
    image_shape = image.shape[:2]  # h, w

    # To ensure bbox always remap to original image size, we reset image shape here as this is only preprocessing
    sample[DataKeys.IMAGE_HEIGHT] = image_shape[0]
    sample[DataKeys.IMAGE_WIDTH] = image_shape[1]

    if "objects" in sample:  # convert the objects to the new image size
        # here objects are transformed from XYWH_ABS to XYXY_ABS
        sample["objects"] = [
            transform_instance_annotations(obj, geometric_tf, image_shape)
            for obj in sample["objects"]
            if obj.get("iscrowd", 0) == 0
        ]

    # update image in place
    sample[DataKeys.IMAGE] = image
    return sample


def detectron2_realtime_ransform_image_and_objects(sample, geometric_tf, mask_on):
    from detectron2.data.detection_utils import (
        annotations_to_instances,
        filter_empty_instances,
        transform_instance_annotations,
    )
    from detectron2.data.transforms import apply_transform_gens
    from detectron2.structures import BoxMode

    # we always read image in RGB format in the dataset, when it reaches here the image is of numpay array with shape (h, w, c)
    # detectron2 needs image of shape (h, w, c) and in this place of transformation.
    # here we once resize the image to max possible sizes needed during training/testing
    image = np.array(sample[DataKeys.IMAGE])

    # sample must contain image height and image width as done for coco type datasets
    # here we assume that the sample has those. If not the image width and heights are set
    check_image_size(sample, image)

    # here the image is resized to correct aspect ratio
    # the returned geometric_tf is needed for bbox transformation in detectron2
    image, geometric_tf = apply_transform_gens(geometric_tf, image)

    # store the image shape here
    image_shape = image.shape[:2]  # h, w

    if "objects" in sample:  # convert the objects to the new image size
        # USER: Modify this if you want to keep them for some reason.
        for obj in sample["objects"]:
            if not mask_on:
                obj.pop("segmentation", None)
            obj.pop("keypoints", None)

            if "bbox_mode" in obj:
                obj["bbox_mode"] = BoxMode(obj["bbox_mode"])

        # here objects are transformed from XYWH_ABS to XYXY_ABS
        sample["objects"] = [
            transform_instance_annotations(obj, geometric_tf, image_shape)
            for obj in sample["objects"]
            if obj.get("iscrowd", 0) == 0
        ]

        instances = annotations_to_instances(sample["objects"], image_shape)
        sample["instances"] = filter_empty_instances(instances)

    # convert image to tensor and update in place
    sample[DataKeys.IMAGE] = (
        torch.as_tensor(  # here image is kept as 0-255 as that is what is required in detectron2
            np.ascontiguousarray(image.astype("float32").transpose(2, 0, 1))
        )
    )
    return sample


@dataclass
class PILNumpyResize:
    rescale_size: Optional[List[int]] = None

    def __call__(self, image) -> Any:
        if isinstance(image, PIL.Image.Image):
            return image.resize(self.rescale_size)
        elif isinstance(image, np.ndarray):
            return PIL.Image.fromarray(image).resize(self.rescale_size)
        else:
            raise NotImplementedError()


@dataclass
class PILEncode:
    encode_format: str = "PNG"

    def __call__(self, image) -> Any:
        if isinstance(image, np.ndarray):
            image = PIL.Image.fromarray(image)

        buffer = BytesIO()
        image.save(buffer, format=self.encode_format)
        return buffer.getvalue()


@dataclass
class Binarize:
    binarize_threshold: float = 0.5

    def __call__(self, image) -> Any:
        return (image > self.binarize_threshold).to(image.dtype)


@dataclass
class To3ChannelGray:
    def __call__(self, image) -> Any:
        return image.convert("L").convert("RGB")


@dataclass
class ImageSelect:
    index: Union[int, List[int]] = 0
    random_select: bool = False

    def __call__(self, image) -> Any:
        if self.random_select:
            images = []
            images.append(image[0])

            rand_index = random.randint(1, len(image) - 1)
            images.append(image[rand_index])
            return images
        else:
            if isinstance(self.index, list):
                return [image[idx] for idx in self.index]
            else:
                return image[self.index]


@dataclass
class ImagePreprocess(DataAugmentation):
    """
    Defines a basic image preprocessing augmentation for image classification.
    """

    square_pad: bool = False
    rescale_size: Optional[List[int]] = None
    encode_image: bool = False
    encode_format: str = "PNG"

    def __str__(self):
        return str(self._aug)

    def _initialize_aug(self):
        # generate transformations list
        aug = []

        # apply square padding if required
        if self.square_pad:
            aug.append(SquarePad())

        # apply rescaling if required
        if self.rescale_size is not None:
            aug.append(PILNumpyResize(rescale_size=self.rescale_size))

        # apply encoding if required
        if self.encode_image:
            aug.append(PILEncode(encode_format=self.encode_format))

        # generate torch transformation
        return transforms.Compose(aug)

    def __post_init__(self):
        self._aug = self._initialize_aug()

    def __call__(self, sample):
        if isinstance(sample, list):
            return [self._aug(s) for s in sample]
        else:
            return self._aug(sample)


@dataclass
class ObjectDetectionImagePreprocess(DataAugmentation):
    """
    Defines a basic image preprocessing for image object detection based on detectron2.
    """

    encode_image: bool = True
    encode_format: str = "PNG"

    # detection related
    min_size: int = (
        800  # this is the size used in detectron2 default config, we preprocess all image to this size
    )
    max_size: int = (
        1333  # this is the size used in detectron2 default config, we preprocess all image to this size
    )

    def _initialize_aug(self):
        from detectron2.data.transforms import ResizeShortestEdge

        # generate transformations list
        self.image_postprocess_tf = transforms.Compose(
            [transforms.ToPILImage(), PILEncode(encode_format=self.encode_format)]
        )

        # this can only be applied to tensors
        self.geometric_tf = [ResizeShortestEdge(self.min_size, self.max_size)]

    def __post_init__(self):
        self._initialize_aug()

    def __call__(self, sample):
        sample = detectron2_preprocess_transform_image_and_objects(
            sample, self.geometric_tf
        )

        if self.encode_image:
            sample[DataKeys.IMAGE] = self.image_postprocess_tf(sample[DataKeys.IMAGE])
        return sample


@dataclass
class ObjectDetectionImageAug(DataAugmentation):
    """
    Defines a basic image augmentation for image object detection based on detectron2.
    """

    # detection related
    min_size: Union[int, List[int]] = field(
        default_factory=lambda: [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    )
    max_size: int = (
        1333  # this is the size used in detectron2 default config, we preprocess all image to this size
    )
    mask_on: bool = False
    random_flip: bool = field(
        default=False, metadata={"help": "Whether to perform random horizontal flip."}
    )
    sampling_style: str = "choice"
    keep_objects: bool = False

    def _initialize_aug(self):
        from detectron2.data.transforms import RandomFlip, ResizeShortestEdge

        # this can only be applied to tensors
        self.geometric_tf = []
        if self.random_flip:
            self.geometric_tf.append(RandomFlip())
        self.geometric_tf.append(
            ResizeShortestEdge(
                self.min_size, self.max_size, sample_style=self.sampling_style
            )
        )

    def __post_init__(self):
        self._initialize_aug()

    def __call__(self, sample):
        sample = detectron2_realtime_ransform_image_and_objects(
            sample, self.geometric_tf, mask_on=self.mask_on
        )

        if not self.keep_objects:
            sample.pop("objects", None)
        return sample


@dataclass
class ObjDetectionBasicAug(DataAugmentation):
    """
    Defines a basic image augmentation for image object detection.
    """

    rgb_to_bgr: bool = False
    rescale_strategy: Optional[ClassInitializerArgs] = field(
        default=None,
    )
    normalize: bool = True
    random_hflip: bool = field(
        default=False, metadata={"help": "Whether to perform random horizontal flip."}
    )
    random_vflip: bool = field(
        default=False, metadata={"help": "Whether to perform random vertical flip."}
    )

    # detection related
    mask_on: bool = False

    def __str__(self):
        return str(self._aug)

    def _initialize_aug(self):
        # create rescaling transform
        self.rescale_transform = None
        if self.rescale_strategy is not None:
            self.rescale_transform = DataAugmentationFactory.create(
                name=self.rescale_strategy.name, kwargs=self.rescale_strategy.kwargs
            )

        # generate transformations list
        aug = []

        # convert images to tensor
        aug.append(transforms.ToTensor())

        # apply rgb to bgr if required
        if self.rgb_to_bgr:
            aug.append(RGBToBGR())

        # apply rescaling if required
        if self.rescale_transform is not None:
            aug.append(self.rescale_transform)

        # apply random horizontal flip if required
        if self.random_hflip:
            aug.append(transforms.RandomHorizontalFlip(0.5))

        # apply random vertical flip if required
        if self.random_vflip:
            aug.append(transforms.RandomVerticalFlip(0.5))

        # change dtype to float
        aug.append(transforms.ConvertImageDtype(torch.float))

        # normalize image if required
        if self.normalize:
            if isinstance(self.mean, float):
                aug.append(transforms.Normalize((self.mean,), (self.std,)))
            else:
                aug.append(transforms.Normalize(self.mean, self.std))

        # generate torch transformation
        return transforms.Compose(aug)

    def __post_init__(self):
        self._aug = self._initialize_aug()

    def __call__(self, sample):
        if not isinstance(sample, list):
            samples = [sample]

        for sample in samples:
            # image sanity check
            image = sample[DataKeys.IMAGE]  # PIL Image with (W, H, C)

            # sample has width, height information
            check_image_size(sample, image)

            image = self._aug(image)

            # handle objects
            objects = sample.get("objects", None)
            for object in objects:
                if not self.mask_on:
                    object.pop("segmentation", None)
                object.pop("keypoints", None)

            annos = [
                transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)


@dataclass
class Cifar10Aug(DataAugmentation):
    """
    Defines a basic image augmentation for CIFAR10 dataset classification.
    """

    mean: Union[float, List[float]] = (0.4914, 0.4822, 0.4465)
    std: Union[float, List[float]] = (0.247, 0.243, 0.261)
    pad_size: int = 4
    crop_size: int = 32
    train: bool = False

    def __str__(self):
        return str(self._aug)

    def _initialize_aug(self):
        if self.train:
            return transforms.Compose(
                [
                    transforms.Pad(self.pad_size, padding_mode="reflect"),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(self.crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize(np.array(self.mean), np.array(self.std)),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(np.array(self.mean), np.array(self.std)),
                ]
            )

    def __post_init__(self):
        self._aug = self._initialize_aug()

    def __call__(self, sample):
        if isinstance(sample, list):
            return [self._aug(s) for s in sample]
        else:
            return self._aug(sample)


@dataclass
class BasicImageAug(DataAugmentation):
    """
    Defines a basic image augmentation for image classification.
    """

    gray_to_rgb: bool = False
    rgb_to_bgr: bool = False
    rgb_to_gray: bool = False
    rescale_strategy: Optional[ClassInitializerArgs] = field(
        default=None,
    )
    center_crop: Optional[List[int]] = None
    normalize: bool = True
    random_hflip: bool = field(
        default=False, metadata={"help": "Whether to perform random horizontal flip."}
    )
    random_vflip: bool = field(
        default=False, metadata={"help": "Whether to perform random vertical flip."}
    )
    mean: Union[float, List[float]] = (0.485, 0.456, 0.406)
    std: Union[float, List[float]] = (0.229, 0.224, 0.225)
    add_gaussian_noise: float = 0.0
    binarize: bool = False
    binarize_threshold: float = 0.5
    to_3_channel_gray: bool = False
    image_select: Optional[Union[str, int, List[int]]] = None

    def __str__(self):
        return str(self._aug)

    def _initialize_aug(self):
        # create rescaling transform
        self.rescale_transform = None
        if self.rescale_strategy is not None:
            self.rescale_transform = DataAugmentationFactory.create(
                name=self.rescale_strategy.name, kwargs=self.rescale_strategy.kwargs
            )

        # generate transformations list
        aug = []

        self.selection_aug = None
        if self.image_select is not None:
            if self.image_select == "random":
                self.selection_aug = ImageSelect(index=0, random_select=True)
            else:
                self.selection_aug = ImageSelect(index=self.image_select)

        # apply rgb to bgr if required
        if self.rgb_to_gray:
            aug.append(RGBToGrayScale())

        if self.to_3_channel_gray:
            aug.append(To3ChannelGray())

        # convert images to tensor
        aug.append(transforms.ToTensor())

        # apply gray to rgb if required
        if self.gray_to_rgb:
            aug.append(GrayScaleToRGB())

        # apply rgb to bgr if required
        if self.rgb_to_bgr:
            aug.append(RGBToBGR())

        # apply rescaling if required
        if self.rescale_transform is not None:
            aug.append(self.rescale_transform)

        # apply image binarization if required
        if self.binarize:
            aug.append(Binarize(binarize_threshold=self.binarize_threshold))

        # apply center crop if required
        if self.center_crop is not None:
            aug.append(transforms.CenterCrop(self.center_crop))

        # apply random horizontal flip if required
        if self.random_hflip:
            aug.append(transforms.RandomHorizontalFlip(0.5))

        # apply random vertical flip if required
        if self.random_vflip:
            aug.append(transforms.RandomVerticalFlip(0.5))

        # change dtype to float
        aug.append(transforms.ConvertImageDtype(torch.float))

        # if add add_gaussian_noise
        if self.add_gaussian_noise > 0:
            aug.append(
                transforms.RandomApply(
                    [GaussianNoiseRGB(magnitude=self.add_gaussian_noise)], p=0.5
                )
            )

        # normalize image if required
        if self.normalize:
            if isinstance(self.mean, float):
                aug.append(transforms.Normalize((self.mean,), (self.std,)))
            else:
                aug.append(transforms.Normalize(self.mean, self.std))

        # generate torch transformation
        return transforms.Compose(aug)

    def __post_init__(self):
        self._aug = self._initialize_aug()

    def __call__(self, sample):
        if self.selection_aug is not None:
            sample = self.selection_aug(sample)
        if isinstance(sample, list):
            return [self._aug(s) for s in sample]
        else:
            return self._aug(sample)


@dataclass
class BinarizationAug(DataAugmentation):
    """
    Defines a image augmentation for image to image binarizaiton task .
    """

    gray_to_rgb: bool = False
    rgb_to_gray: bool = False
    normalize: bool = True
    random_hflip: bool = field(
        default=False, metadata={"help": "Whether to perform random horizontal flip."}
    )
    random_vflip: bool = field(
        default=False, metadata={"help": "Whether to perform random vertical flip."}
    )
    mean: Union[float, List[float]] = (0.5, 0.5, 0.5)
    std: Union[float, List[float]] = (0.5, 0.5, 0.5)
    image_size: int = 256

    def __str__(self):
        return str(self._aug)

    def _initialize_aug(self):
        aug = []
        if self.rgb_to_gray:
            aug.append(RGBToGrayScale())
        aug.append(transforms.ToTensor())
        if self.gray_to_rgb:
            aug.append(GrayScaleToRGB())
        aug.append(transforms.ConvertImageDtype(torch.float))
        if self.normalize:
            if isinstance(self.mean, float):
                aug.append(transforms.Normalize((self.mean,), (self.std,)))
            else:
                aug.append(transforms.Normalize(self.mean, self.std))
        return transforms.Compose(aug)

    def _initialize_aug_gt(self):
        aug = []
        aug.append(RGBToGrayScale())
        aug.append(transforms.ToTensor())
        aug.append(Binarize())
        aug.append(transforms.ConvertImageDtype(torch.float))
        if self.normalize:
            if isinstance(self.mean, float):
                aug.append(transforms.Normalize((self.mean,), (self.std,)))
            else:
                aug.append(transforms.Normalize(self.mean, self.std))
        return transforms.Compose(aug)

    def __post_init__(self):
        self._aug = self._initialize_aug()
        self._aug_gt = self._initialize_aug_gt()

    def __call__(self, sample):
        image = sample[DataKeys.IMAGE]
        gt_image = sample[DataKeys.GT_IMAGE]

        import torchvision.transforms.functional as TF

        # random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            gt_image, output_size=(self.image_size, self.image_size)
        )
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(self.image_size, self.image_size)
        )
        image = TF.crop(image, i, j, h, w)
        gt_image = TF.crop(gt_image, i, j, h, w)

        # random horizontal flipping
        if self.random_hflip and random.random() > 0.5:
            gt_image = TF.hflip(gt_image)
            image = TF.hflip(image)

        # random vertical flipping
        if self.random_vflip and random.random() > 0.5:
            gt_image = TF.vflip(gt_image)
            image = TF.vflip(image)

        image = self._aug(image)
        gt_image = self._aug_gt(gt_image)

        return {DataKeys.IMAGE: image, DataKeys.GT_IMAGE: gt_image}


@dataclass
class RandAug(DataAugmentation):
    """
    Applies the ImageNet Random Augmentation to torch tensor or numpy image as
    defined in timm for image classification with little modification.
    """

    input_size: int = 224
    is_training: bool = True
    use_prefetcher: bool = False
    no_aug: bool = False
    scale: Optional[float] = None
    ratio: Optional[float] = None
    hflip: float = 0.5
    vflip: float = 0.0
    color_jitter: Union[float, List[float]] = 0.4
    auto_augment: Optional[str] = "rand-m9-mstd0.5-inc1"
    interpolation: str = "bicubic"
    mean: Union[float, List[float]] = (0.485, 0.456, 0.406)
    std: Union[float, List[float]] = (0.229, 0.224, 0.225)
    re_prob: float = 0.0
    re_mode: str = "const"
    re_count: int = 1
    re_num_splits: int = 0
    crop_pct: Optional[float] = None
    tf_preprocessing: bool = False
    separate: bool = False

    # custom args
    n_augs: int = 1
    fixed_resize: bool = False

    def __str__(self):
        if self.n_augs == 1:
            return str(self._aug)
        else:
            return str([self._aug for _ in self.n_augs])

    def __post_init__(self):
        self._aug = self._initialize_aug()

    def _initialize_aug(self):
        torch_str_to_interpolation = {
            "nearest": transforms.InterpolationMode.NEAREST,
            "bilinear": transforms.InterpolationMode.BILINEAR,
            "bicubic": transforms.InterpolationMode.BICUBIC,
            "box": transforms.InterpolationMode.BOX,
            "hamming": transforms.InterpolationMode.HAMMING,
            "lanczos": transforms.InterpolationMode.LANCZOS,
        }

        aug = create_transform(
            input_size=self.input_size,
            is_training=self.is_training,
            use_prefetcher=self.use_prefetcher,
            no_aug=self.no_aug,
            scale=self.scale,
            ratio=self.ratio,
            hflip=self.hflip,
            vflip=self.vflip,
            color_jitter=self.color_jitter,
            auto_augment=self.auto_augment,
            interpolation=self.interpolation,
            mean=self.mean,
            std=self.std,
            re_prob=self.re_prob,
            re_mode=self.re_mode,
            re_count=self.re_count,
            re_num_splits=self.re_num_splits,
            crop_pct=self.crop_pct,
            tf_preprocessing=self.tf_preprocessing,
            separate=self.separate,
        ).transforms

        # replace random resized crop with fixed resizing if required
        if self.fixed_resize:
            aug[0] = transforms.Resize(
                self.input_size,
                interpolation=torch_str_to_interpolation[self.interpolation],
            )

            # this makes sure image is always 3-channeled.
            aug.insert(0, GrayScaleToRGB())

            # this makes sure image is always 3-channeled.
            aug.insert(2, transforms.ToPILImage())
        else:
            # this makes sure image is always 3-channeled.
            aug.insert(0, GrayScaleToRGB())

            # this makes sure image is always 3-channeled.
            aug.insert(1, transforms.ToPILImage())

        # generate torch transformation
        return transforms.Compose(aug)

    def __call__(self, image: torch.Tensor):
        if self.n_augs == 1:
            return self._aug(image)
        else:
            augs = []
            for _ in self.n_augs:
                augs.append(self._aug(image))
                augs.append(self._aug(image))
        return augs


@dataclass
class Moco(DataAugmentation):
    """
    Applies the Standard Moco Augmentation to a torch tensor image.
    """

    image_size: Union[int, Tuple[int, int]] = 224
    gray_to_rgb: bool = False
    to_pil: bool = False
    mean: Union[float, List[float]] = (0.485, 0.456, 0.406)
    std: Union[float, List[float]] = (0.229, 0.224, 0.225)

    def __str__(self):
        return str([self.aug, self.aug])

    def __post_init__(self):
        self._aug = self._initialize_aug()

    def _initialize_aug(self):
        base_aug = []
        if self.gray_to_rgb:
            base_aug.append(GrayScaleToRGB())

        if self.to_pil:
            base_aug.append(transforms.ToPILImage())

        return transforms.Compose(
            base_aug
            + [
                transforms.RandomResizedCrop(
                    self.image_size, scale=(0.2, 1.0), interpolation=PILImage.BICUBIC
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlurPIL(sigma=(0.1, 2.0))], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def __call__(self, image):
        crops = []
        crops.append(self._aug(image))
        crops.append(self._aug(image))
        return crops


@dataclass
class BarlowTwins(DataAugmentation):
    """
    Applies the Standard BarlowTwins Augmentation to a torch tensor image.
    """

    image_size: Union[int, Tuple[int, int]] = 224
    gray_to_rgb: bool = False
    to_pil: bool = False
    mean: Union[float, List[float]] = (0.485, 0.456, 0.406)
    std: Union[float, List[float]] = (0.229, 0.224, 0.225)

    def __str__(self):
        return str([self.aug, self.aug])

    def __post_init__(self):
        self._aug = self._initialize_aug()

    def _initialize_aug(self):
        base_aug = []
        if self.gray_to_rgb:
            base_aug.append(GrayScaleToRGB())
        if self.to_pil:
            base_aug.append(transforms.ToPILImage())

        aug1 = transforms.Compose(
            base_aug
            + [
                transforms.RandomResizedCrop(
                    self.image_size, interpolation=PILImage.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlurPIL([0.1, 2.0])], p=1.0),
                transforms.RandomApply([Solarization()], p=0.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )
        aug2 = transforms.Compose(
            base_aug
            + [
                transforms.RandomResizedCrop(
                    self.image_size, interpolation=PILImage.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlurPIL([0.1, 2.0])], p=0.1),
                transforms.RandomApply([Solarization()], p=0.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

        return aug1, aug2

    def __call__(self, image):
        crops = []
        crops.append(self._aug[0](image))
        crops.append(self._aug[1](image))
        return crops


@dataclass
class MultiCrop(DataAugmentation):
    """
    Applies the Standard Multicrop Augmentation to a torch tensor image.
    """

    image_size: Union[int, Tuple[int, int]] = 224
    gray_to_rgb: bool = False
    to_pil: bool = False
    mean: Union[float, List[float]] = (0.485, 0.456, 0.406)
    std: Union[float, List[float]] = (0.229, 0.224, 0.225)
    global_crops_scale: Tuple[float, float] = (0.4, 1.0)
    local_crops_scale: Tuple[float, float] = (0.05, 0.4)
    local_crops_number: int = 12
    local_crop_size: Union[int, Tuple[int, int]] = 96

    def __str__(self):
        return json.dumps(
            {
                "global_1": self.global_transform_1,
                "global_2": self.global_transform_2,
                "local": self.local_transform,
            },
            indent=2,
        )

    def __post_init__(self):
        self._aug = self._initialize_aug()

    def _initialize_aug(self):
        base_aug = []
        if self.gray_to_rgb:
            base_aug.append(GrayScaleToRGB())
        if self.to_pil:
            base_aug.append(transforms.ToPILImage())

        flip_and_color_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )
        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

        # first global crop
        self.global_transform_1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.image_size,
                    scale=self.global_crops_scale,
                    interpolation=PILImage.BICUBIC,
                ),
                flip_and_color_jitter,
                transforms.RandomApply([GaussianBlurPIL([0.1, 2.0])], p=1.0),
                normalize,
            ]
        )
        # second global crop
        self.global_transform_2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.image_size,
                    scale=self.global_crops_scale,
                    interpolation=PILImage.BICUBIC,
                ),
                flip_and_color_jitter,
                transforms.RandomApply([GaussianBlurPIL([0.1, 2.0])], p=0.1),
                transforms.RandomApply([Solarization()], p=0.2),
                normalize,
            ]
        )
        # transformation for the local small crops
        self.local_crops_number = self.local_crops_number
        self.local_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.local_crop_size,
                    scale=self.local_crops_scale,
                    interpolation=PILImage.BICUBIC,
                ),
                flip_and_color_jitter,
                transforms.RandomApply([GaussianBlurPIL([0.1, 2.0])], p=0.5),
                normalize,
            ]
        )

        return None

    def __call__(self, image: torch.Tensor):
        crops = []
        crops.append(self.global_transform_1(image))
        crops.append(self.global_transform_2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))
        return crops


#
# @dataclass
# class TwinDocs(DataAugmentation):
#     """
#     Applies the TwinDocs Augmentation to a torch tensor image
#     (experimental augmentation for document images).
#     """

#     image_size: Union[int, Tuple[int, int]] = 224
#     gray_to_rgb: bool = False
#     to_pil: bool = False
# mean: Union[float, List[float]] = (0.485, 0.456, 0.406)
# std: Union[float, List[float]] = (0.229, 0.224, 0.225)

#     def __post_init__(self):
#         base_aug = []
#         if self.gray_to_rgb:
#             base_aug.append(GrayScaleToRGB())
#         if self.to_pil:
#             base_aug.append(transforms.ToPILImage())
#         self.aug1 = transforms.Compose(
#             [
#                 RandomResizedMaskedCrop(self.image_size, (0.4, 0.8)),
#                 *base_aug,
#                 transforms.RandomApply(
#                     [transforms.RandomAffine((-2, 2), fill=255)], p=0.5
#                 ),
#                 transforms.RandomApply(
#                     [transforms.RandomAffine(0, translate=(0.2, 0.2), fill=255)], p=0.5
#                 ),
#                 transforms.RandomApply(
#                     [transforms.RandomAffine(0, scale=(0.9, 1.0), fill=255)], p=0.5
#                 ),
#                 transforms.RandomApply(
#                     [transforms.RandomAffine(0, shear=(-2, 2), fill=255)], p=0.5
#                 ),
#                 # transforms.RandomHorizontalFlip(p=0.5),
#                 transforms.RandomApply(
#                     [
#                         transforms.ColorJitter(
#                             brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
#                         )
#                     ],
#                     p=0.8,
#                 ),
#                 transforms.RandomGrayscale(p=0.2),
#                 transforms.RandomApply([GaussianBlurPIL([0.1, 0.5])], p=1.0),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=self.mean, std=self.std),
#             ]
#         )
#         self.aug2 = transforms.Compose(
#             [
#                 RandomResizedMaskedCrop(self.image_size, (0.4, 0.8)),
#                 *base_aug,
#                 transforms.RandomApply(
#                     [transforms.RandomAffine((-5, 5), fill=255)], p=0.5
#                 ),
#                 transforms.RandomApply(
#                     [transforms.RandomAffine(0, translate=(0.2, 0.2), fill=255)], p=0.5
#                 ),
#                 transforms.RandomApply(
#                     [transforms.RandomAffine(0, scale=(0.9, 1.0), fill=255)], p=0.5
#                 ),
#                 transforms.RandomApply(
#                     [transforms.RandomAffine(0, shear=(-5, 5), fill=255)], p=0.5
#                 ),
#                 # transforms.RandomHorizontalFlip(p=0.5),
#                 transforms.RandomApply(
#                     [
#                         transforms.ColorJitter(
#                             brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
#                         )
#                     ],
#                     p=0.8,
#                 ),
#                 transforms.RandomGrayscale(p=0.2),
#                 transforms.RandomApply([GaussianBlurPIL([0.1, 0.5])], p=1.0),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=self.mean, std=self.std),
#             ]
#         )

#     def __call__(self, image):
#         crops = []
#         crops.append(self.aug1(image))
#         crops.append(self.aug2(image))
#         return crops
