from __future__ import annotations

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

from torchfusion.core.constants import DataKeys


class DataVisualizationMixin:
    def show_images(self, batch, nmax=16, show=True):
        image_grid = make_grid((batch[:nmax]), nrow=4)
        if show:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xticks([])
            ax.set_yticks([])
            print(image_grid.shape, image_grid.permute(1, 2, 0).shape)
            ax.imshow(image_grid.permute(1, 2, 0))
            plt.show()
            plt.close(fig)
        return image_grid

    def show_batch(self, batch):
        draw_batch = []
        draw_batch_gt = []
        batch = [dict(zip(batch, t)) for t in zip(*batch.values())]
        for sample in batch:
            image = sample[DataKeys.IMAGE].permute(1, 2, 0).cpu().numpy()
            if DataKeys.GT_IMAGE in sample:
                gt_image = sample[DataKeys.GT_IMAGE].permute(1, 2, 0).cpu().numpy()
                gt_image = np.ascontiguousarray(gt_image)
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
                draw_batch.append(torch.from_numpy(image).permute(2, 0, 1))
                draw_batch_gt.append(torch.from_numpy(gt_image).permute(2, 0, 1))
            except:
                pass

        # draw images
        if len(draw_batch) > 0:
            print("drawing")
            self.show_images(draw_batch, show=True)
        if len(draw_batch_gt) > 0:
            self.show_images(draw_batch_gt, show=True)
