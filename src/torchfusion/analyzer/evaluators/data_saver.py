from pathlib import Path
from typing import List, Mapping

import h5py
import numpy as np
import torch
from ignite.engine import Engine

from torchfusion.analyzer.evaluators.image_reconstruction_evaluator import EvaluatorBase
from torchfusion.analyzer.evaluators.utilities import update_dataset_at_indices
from torchfusion.core.constants import DataKeys


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

        for evaluator in self._attached_evaluators:
            evaluator.write_data_to_hdf5(engine, hf, evaluator.name, indices)

        for key in self._keys_to_save:
            # add labels
            self.add_key_from_output(engine, hf, key, indices)

        hf.close()
