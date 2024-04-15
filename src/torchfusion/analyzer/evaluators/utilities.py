"""
Defines the feature attribution generation task.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List

import h5py
import numpy as np
import torch
from ignite.engine import Engine

from torchfusion.core.constants import DataKeys
from torchfusion.utilities.logging import get_logger

if TYPE_CHECKING:
    from torchfusion.analyzer.evaluators.evaluator_base import EvaluatorBase


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


class DataSaverHandler:
    def __init__(
        self,
        output_file,
        attached_evaluators: List[EvaluatorBase],
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
