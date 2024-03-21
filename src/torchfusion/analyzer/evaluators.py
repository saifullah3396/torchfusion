"""
Defines the feature attribution generation task.
"""

import abc
import math
from abc import abstractmethod
from pathlib import Path
from typing import List, Mapping, Union

import h5py
import numpy as np
import torch
import torchvision
from ignite.engine import CallableEventWithFilter, Engine, EventEnum, Events, EventsList
from torch.utils.tensorboard import SummaryWriter

from torchfusion.core.constants import DataKeys
from torchfusion.utilities.logging import get_logger


def update_dataset_at_indices(
    hf: h5py.File,
    key: str,
    indices: np.array,
    data,
    maxshape=(None,),
    overwrite: bool = False,
):
    if key not in hf:
        hf.create_dataset(
            key, data=data, compression="gzip", chunks=True, maxshape=maxshape
        )
    else:
        if maxshape[1:] != hf[key].shape[1:]:
            logger = get_logger()
            if overwrite:
                logger.info(
                    f"Reinitializing data due to shape mismatch for key={key} since overwrite is set to True."
                )
                del hf[key]
                hf.create_dataset(
                    key, data=data, compression="gzip", chunks=True, maxshape=maxshape
                )
            else:
                logger.error(
                    f"Data overwrite is set to False but there is mismatch between data shapes for key = {key}"
                )
                exit()

        max_len = indices.max() + 1
        if len(hf[key]) < max_len:
            hf[key].resize((indices.max() + 1), axis=0)
            hf[key][indices] = data
        elif overwrite:
            hf[key][indices] = data


class EvaluatorKeys:
    IMAGE_RECONSTRUCTION = "image_reconstruction"


class EvaluatorEvents(EventEnum):
    IMAGE_RECONSTRUCTION_COMPUTED = f"{EvaluatorKeys.IMAGE_RECONSTRUCTION}_computed"


class EvaluatorBase(abc.ABC):
    def __init__(
        self, output_file: str, summary_writer: SummaryWriter, overwrite: bool = False
    ):
        self._output_file = output_file
        self._summary_writer = summary_writer
        self._overwrite = overwrite

    @abstractmethod
    def compute(
        self,
        engine: Engine,
    ):
        pass

    @abstractmethod
    def fire_completion_event(self, engine: Engine):
        pass

    def write_data_to_hdf5(
        self, engine: Engine, hf: h5py.File, key: str, indices: np.ndarray
    ):
        data = getattr(engine.state, key)
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        update_dataset_at_indices(
            hf, key, indices, data, (None, *data.shape[1:]), overwrite=self._overwrite
        )

    def read_data_from_hdf5(self, engine: Engine, name: str, output_file: str):
        # read from h5 if this data is already computed
        if Path(output_file).exists():
            hf = h5py.File(output_file, "r")

            # get data indices
            indices = np.array(engine.state.batch["index"])

            # check if required data is already computed
            if name in hf and indices.max() < len(hf[name]):
                data = self.transform_data_on_load(hf[name][indices])

                # close file
                hf.close()

                return data

    def transform_data_on_load(self, data):
        return data

    def __call__(self, engine: Engine, name: str) -> None:
        data = None
        if not self._overwrite:  # do not read data if overwrite is true
            data = self.read_data_from_hdf5(engine, name, self._output_file)
        if data is None:
            data = self.compute(engine)
        setattr(engine.state, name, data)
        self.fire_completion_event(engine)

    def attach(
        self,
        engine: Engine,
        name: str = "attr_map",
        event: Union[
            str, Events, CallableEventWithFilter, EventsList
        ] = Events.ITERATION_COMPLETED,
    ) -> None:
        if not hasattr(engine.state, name):
            setattr(engine.state, name, None)
        engine.add_event_handler(event, self, name)


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
        self._output_transform = lambda x: x[DataKeys.IMAGE_RECONS]
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
                DataKeys.IMAGE_RECONS,
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
        engine.fire_event(EvaluatorEvents.IMAGE_RECONSTRUCTION_COMPUTED)

    def write_data_to_hdf5(
        self, engine: Engine, hf: h5py.File, key: str, indices: np.array
    ):
        # add labels
        data = getattr(engine.state, EvaluatorKeys.IMAGE_RECONSTRUCTION)
        if EvaluatorKeys.IMAGE_RECONSTRUCTION not in hf:
            base_group = hf.create_group(EvaluatorKeys.IMAGE_RECONSTRUCTION)
        else:
            base_group = hf[EvaluatorKeys.IMAGE_RECONSTRUCTION]
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


class DataSaverHandler:
    def __init__(
        self,
        output_file,
        attached_evaluators: Mapping[str, EvaluatorBase],
        keys_to_save=List[str],
    ):
        self._attached_evaluators = attached_evaluators
        self._output_file = output_file
        self._keys_to_save = (
            [DataKeys.LABEL, DataKeys.PRED, DataKeys.IMAGE_FILE_PATH]
            if keys_to_save is None
            else keys_to_save
        )

    def add_key_from_output(
        self, engine: Engine, hf: h5py.File, key: str, indices: np.array
    ):
        # add labels
        if key in engine.state.output:
            data = engine.state.output[key]
        else:
            data = engine.state.batch[key]
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        max_shape = (None,)
        if isinstance(data, np.ndarray):
            max_shape = (None, *data.shape[1:])
        elif isinstance(data, list):
            max_shape = (None,)
        update_dataset_at_indices(
            hf, key=key, indices=indices, data=data, maxshape=max_shape, overwrite=True
        )

    def __call__(self, engine: Engine) -> None:
        # this part is quite slow? maybe we can speed up by not overwriting but just leaving already written indces?
        if not Path(self._output_file).parent.exists():
            Path(self._output_file).parent.mkdir(parents=True)

        hf = h5py.File(self._output_file, "a")

        # get data indices
        indices = engine.state.batch["index"]
        indices = np.array(indices)

        # create index dataset
        update_dataset_at_indices(hf, key="index", indices=indices, data=indices)

        for key, evaluator in self._attached_evaluators.items():
            evaluator.write_data_to_hdf5(engine, hf, key, indices)

        for key in self._keys_to_save:
            # add labels
            self.add_key_from_output(engine, hf, key, indices)

        hf.close()
