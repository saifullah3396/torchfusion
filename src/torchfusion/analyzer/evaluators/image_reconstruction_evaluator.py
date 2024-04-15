"""
Defines the feature attribution generation task.
"""

import math
from pathlib import Path

import h5py
import numpy as np
import torch
import torchvision
from ignite.engine import Engine
from torch.utils.tensorboard import SummaryWriter

from torchfusion.analyzer.evaluators.evaluator_base import EvaluatorBase
from torchfusion.analyzer.evaluators.utilities import update_dataset_at_indices
from torchfusion.core.constants import DataKeys


class ImageReconstructionEvaluator(EvaluatorBase):
    def __init__(
        self,
        output_file: str,
        summary_writer: SummaryWriter,
        overwrite=False,
    ):
        super().__init__(
            output_file=output_file, summary_writer=summary_writer, overwrite=overwrite
        )
        self._output_transform = lambda x: x[DataKeys.RECONS]
        self.images_saved = False

    def compute(self, engine: Engine) -> None:
        from torch.nn import functional as F

        image = engine.state.batch[DataKeys.IMAGE]
        reconstruction = self._output_transform(engine.state.output)

        image = (image / 2 + 0.5).clamp(0, 1).detach().cpu()
        reconstruction = (reconstruction / 2 + 0.5).clamp(0, 1).detach().cpu()

        if not self.images_saved:
            self._summary_writer.add_image(
                DataKeys.IMAGE,
                torchvision.utils.make_grid(image, nrow=int(math.sqrt(image.shape[0]))),
            )
            self._summary_writer.add_image(
                DataKeys.RECONS,
                torchvision.utils.make_grid(
                    reconstruction, nrow=int(math.sqrt(image.shape[0]))
                ),
            )
            self.images_saved = True

        # from matplotlib import pyplot as plt

        # plt.imshow(image[0].permute(1, 2, 0).numpy())
        # plt.show()
        # plt.imshow(reconstruction[0].permute(1, 2, 0).numpy())
        # plt.show()
        # exit()

        loss = []
        for i in range(len(image)):
            loss.append(F.mse_loss(image[i], reconstruction[i]))
        return dict(
            loss=loss, image=image.numpy(), recons=reconstruction.detach().cpu().numpy()
        )

    def fire_completion_event(self, engine: Engine):
        engine.fire_event(self.completion_event)

    def write_data_to_hdf5(
        self, engine: Engine, hf: h5py.File, key: str, indices: np.array
    ):
        # add labels
        data = getattr(engine.state, __class__.__name__)
        if self.name not in hf:
            base_group = hf.create_group(self.name)
        else:
            base_group = hf[self.name]
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            elif isinstance(value, list):
                value = np.array(value)
            update_dataset_at_indices(
                base_group,
                key=key,
                indices=indices,
                data=value,
                maxshape=(None, *value.shape[1:]),
                overwrite=self._overwrite,
            )

    def read_data_from_hdf5(self, engine: Engine, name: str, output_file: str):
        # read from h5 if this data is already computed
        if Path(output_file).exists():
            hf = h5py.File(output_file, "r")

            # get data indices
            indices = np.array(engine.state.batch["index"])

            if name in hf:
                output_data = {}
                base_data = hf[name]
                for key, data in base_data.items():
                    data = [dict(zip(data, t)) for t in zip(*data.values())]

                    # check if required data is already computed
                    if indices.max() < len(data):
                        indexed_data = [data[idx] for idx in indices]
                        output_data[key] = indexed_data
                if len(output_data) > 0:
                    return self.transform_data_on_load(output_data)
