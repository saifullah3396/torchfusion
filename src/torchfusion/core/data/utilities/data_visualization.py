from __future__ import annotations

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from transformers import PreTrainedTokenizerBase

from torchfusion.core.constants import DataKeys
from torchfusion.utilities.logging import get_logger


def show_images(batch, nmax=16, show=True):
    image_grid = make_grid((batch[:nmax]), nrow=4)
    _, h, w = image_grid.shape
    if show:
        fig, ax = plt.subplots(figsize=(10, 10 * h / w))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(image_grid.permute(1, 2, 0))
        plt.show()
        plt.close(fig)
    return image_grid


def print_batch_info(batch, tokenizer: PreTrainedTokenizerBase = None):
    logger = get_logger()
    logger.info("Batch information: ")
    if tokenizer is not None:
        logger.info(f"Tokenizer: {tokenizer}")

    for key, value in batch.items():
        if tokenizer is not None and key in [DataKeys.TOKEN_IDS]:
            logger.info(
                f"Batch element={key}, shape={len(value)}, type={type(value[0])}\nExample: {value[0]}"
            )
            logger.info(f"Converted string={tokenizer.decode(token_ids=value[0])}")

        if isinstance(value, (torch.Tensor, np.ndarray)):
            logger.info(
                f"Batch element={key}, shape={value.shape}, type={value.dtype}\nExample: {value[0]}"
            )
        elif isinstance(value, list):
            if isinstance(value[0], (torch.Tensor, np.ndarray)):
                logger.info(
                    f"Batch element={key}, shape={value[0].shape}, type={value[0].dtype}\nExample: {value[0]}"
                )
            else:
                logger.info(
                    f"Batch element={key}, shape={len(value)}, type={type(value[0])}\nExample: {value[0]}"
                )
        else:
            logger.info(f"Batch element={key}, type={type(value)}\nExample: {value}")


def show_batch(batch):
    logger = get_logger()
    draw_batch = []
    draw_batch_gt = []
    batch = [dict(zip(batch, t)) for t in zip(*batch.values())]

    if len(batch) > 4:
        logger.warning(
            "Showing only first 4 images in the batch as high-resolution images may take too much memory..."
        )
        batch = batch[:4]

    for sample in batch:
        image = sample[DataKeys.IMAGE].permute(1, 2, 0).cpu().numpy()
        if DataKeys.GT_IMAGE in sample:
            gt_image = sample[DataKeys.GT_IMAGE].permute(1, 2, 0).cpu().numpy()
            gt_image = np.ascontiguousarray(gt_image)
            draw_batch_gt.append(torch.from_numpy(gt_image).permute(2, 0, 1))
        image = np.ascontiguousarray(image)
        h, w, c = image.shape

        if DataKeys.CAPTION in sample:
            p1 = (w // 4, 20)  # opencv point is (x, y) not (y, x)
            cv2.putText(
                image,
                text=sample[DataKeys.CAPTION],
                org=p1,
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1,
                color=(0, 0, 255),
                thickness=2,
            )

        try:
            if DataKeys.WORDS in sample and DataKeys.WORD_BBOXES in sample:
                for word, box in zip(
                    sample[DataKeys.WORDS], sample[DataKeys.WORD_BBOXES]
                ):  # each box is [x1,y1,x2,y2] normalized
                    p1 = (int(box[0] * w), int(box[1] * h))
                    p2 = (int(box[2] * w), int(box[3] * h))
                    cv2.rectangle(image, p1, p2, (255, 0, 0), 1)
                    cv2.putText(
                        image,
                        text=word,
                        org=p1,
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1,
                        color=(0, 0, 255),
                        thickness=1,
                    )

            if DataKeys.TOKEN_IDS in sample and DataKeys.TOKEN_BBOXES in sample:
                logger.info("Drawing boxes with only first token element on image...")
                last_box = None
                for tokens, box in zip(
                    sample[DataKeys.TOKEN_IDS], sample[DataKeys.TOKEN_BBOXES]
                ):  # each box is [x1,y1,x2,y2] normalized
                    if last_box is not None and last_box == box:
                        continue
                    p1 = (int(box[0] / 1000.0 * w), int(box[1] / 1000.0 * h))
                    p2 = (int(box[2] / 1000.0 * w), int(box[3] / 1000.0 * h))
                    cv2.rectangle(image, p1, p2, (255, 0, 0), 1)
                    cv2.putText(
                        image,
                        text=str(tokens),
                        org=p1,
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1,
                        color=(0, 0, 255),
                        thickness=1,
                    )
                    last_box = box
            draw_batch.append(torch.from_numpy(image).permute(2, 0, 1))
        except Exception as e:
            logger.warning(f"Exception in drawing boxes. Skipping... {e}")

    # draw images
    if len(draw_batch) > 0:
        show_images(draw_batch, show=True)
    if len(draw_batch_gt) > 0:
        show_images(draw_batch_gt, show=True)
