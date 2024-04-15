"""
Defines the feature attribution generation task.
"""

import abc
from abc import abstractmethod
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import torch
from ignite.engine import CallableEventWithFilter, Engine, Events, EventsList
from torch.utils.tensorboard import SummaryWriter

from torchfusion.analyzer.evaluators.utilities import update_dataset_at_indices


class EvaluatorBase(abc.ABC):
    def __init__(
        self, output_file: str, summary_writer: SummaryWriter, overwrite: bool = False
    ):
        self._output_file = output_file
        self._summary_writer = summary_writer
        self._overwrite = overwrite

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def completion_event(self):
        return f"{self.__class__.__name__}_completed"

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
        event: Union[
            str, Events, CallableEventWithFilter, EventsList
        ] = Events.ITERATION_COMPLETED,
    ) -> None:
        if not hasattr(engine.state, self.name):
            setattr(engine.state, self.name, None)
        engine.register_events(self.completion_event)
        engine.add_event_handler(event, self, self.name)
