import numpy as np
import torch
from PIL import Image


def get_images(impath, feature, width, max_height, channels):
    images = []
    if impath:
        images.extend(read_real_images(impath, feature, width, max_height, channels))
    else:
        # do not waste memory for empty images and create 1px height image
        images.append(create_dummy_image(width, max_height, channels))
    return images


def add_images_maybe(self, impath, feature, width, max_height, channels):
    images = self._get_images(impath, feature, width, max_height, channels)

    # simply to single image for usage in this case
    return torch.as_tensor(images[0])


def read_real_images(impath, feature, width: int, max_height: int, channels: int):
    mask = feature.seg_data["pages"]["masks"]
    num_pages = feature.seg_data["pages"]["ordinals"]
    page_sizes = feature.seg_data["pages"]["bboxes"]
    page_sizes = page_sizes[mask].tolist()
    page_lst = num_pages[mask].tolist()
    return [
        get_page_image(impath, page_no, channels, page_size, width)
        for page_no, page_size in zip(page_lst, page_sizes)
    ]


def get_page_image(impath, page_no, channels, height, width):
    page_path = impath / f"{page_no}.png"
    if page_path.is_file():
        return load_image(page_path, channels, height, width)
    else:
        return create_dummy_image(channels, height, width)


def create_dummy_image(channels, height, width) -> np.ndarray:
    arr_sz = (height, width, 3) if channels == 3 else (height, width)
    return np.full(arr_sz, 255, dtype=np.uint8)


def load_image(page_path, channels, height, width):
    image = Image.open(page_path)
    if image.mode != "RGB" and channels == 3:
        image = image.convert("RGB")
    if image.mode != "L" and channels == 1:
        image = image.convert("L")
    return np.array(image.resize((width, height)))
