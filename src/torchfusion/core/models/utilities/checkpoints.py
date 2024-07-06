from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import ignite.distributed as idist
import torch
from fsspec.core import url_to_fs
from fsspec.implementations.local import AbstractFileSystem
from torch import nn
from torchfusion.core.utilities.logging import get_logger

DEFAULT_STATE_DICT_KEY = "state_dict"


def get_filesystem(path: Path, **kwargs: Any) -> AbstractFileSystem:
    fs, _ = url_to_fs(str(path), **kwargs)
    return fs


def load(
    path_or_url: Union[str, Path],
    map_location: Optional[
        Union[
            str,
            Callable,
            torch.device,
            Dict[Union[str, torch.device], Union[str, torch.device]],
        ]
    ] = None,
) -> Any:
    """Loads a checkpoint.

    Args:
        path_or_url: Path or URL of the checkpoint.
        map_location: a function, ``torch.device``, string or a dict specifying how to remap storage locations.
    """
    import torch

    if not isinstance(path_or_url, (str)):
        # any sort of BytesIO or similar
        return torch.load(path_or_url, map_location=map_location)
    if str(path_or_url).startswith("http"):
        return torch.hub.load_state_dict_from_url(
            str(path_or_url), map_location=map_location
        )
    fs = get_filesystem(path_or_url)
    with fs.open(path_or_url, "rb") as f:
        return torch.load(f, map_location=map_location)


def filter_keys(checkpoint, keys: List[str]):
    checkpoint_filtered = {}
    for state in checkpoint:
        updated_state = state
        for key in keys:
            if key in updated_state:
                updated_state = updated_state.replace(key, "")
        checkpoint_filtered[updated_state] = checkpoint[state]
    return checkpoint_filtered


def prepend_keys(checkpoint, keys: List[str]):
    checkpoint_prepended = {}
    for state in checkpoint:
        updated_state = state
        for key in keys:
            if key not in updated_state:
                updated_state = key + updated_state

        checkpoint_prepended[updated_state] = checkpoint[state]
    return checkpoint_prepended


def replace_keys(checkpoint, key: str, replacement: str):
    checkpoint_filtered = {}
    for state in checkpoint:
        updated_state = state
        if key in updated_state:
            updated_state = updated_state.replace(key, replacement)
        checkpoint_filtered[updated_state] = checkpoint[state]
    return checkpoint_filtered


def setup_checkpoint(
    model: nn.Module,
    checkpoint: Optional[str] = None,
    checkpoint_state_dict_key: str = DEFAULT_STATE_DICT_KEY,
    strict: bool = True,
    filtered_keys: Optional[List[str]] = None,
):
    logger = get_logger()

    if not str(checkpoint).startswith("http"):
        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            logger.warning(
                f"Checkpoint not found, cannot load weights from {checkpoint}."
            )
            return

    logger.info(
        f"Loading model from checkpoint file [{checkpoint}] with strict [{strict}]"
    )
    load_from_checkpoint(
        model,
        checkpoint_path=checkpoint,
        checkpoint_state_dict_key=checkpoint_state_dict_key,
        strict=strict,
        filtered_keys=filtered_keys,
    )


def load_checkpoint(checkpoint_path: Union[str, Path]):
    if idist.get_world_size() > 1:
        return load(checkpoint_path, map_location="cpu")
    else:
        return load(checkpoint_path, map_location=idist.device())


def load_from_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    checkpoint_state_dict_key: str,
    strict: bool = True,
    filtered_keys: Optional[List[str]] = None,
):

    # create logger
    logger = get_logger()

    # load the checkpoint from file
    checkpoint = load_checkpoint(checkpoint_path)

    # fine the state_dict_key that is needed to load the model and create a new clean checkpoint
    cleaned_checkpoint = {}

    # check if key is present in checkpoint
    try_default_key = False
    if checkpoint_state_dict_key not in checkpoint:
        # check if any key has been prepended with state_dict_key
        for key in checkpoint:
            if key.startswith(checkpoint_state_dict_key):
                base_key = key.replace(f"{checkpoint_state_dict_key}_", "")
                for key, value in checkpoint[key].items():
                    cleaned_checkpoint[f"{base_key}.{key}"] = value

        if len(cleaned_checkpoint.keys()) == 0:
            logger.warning(
                f"State dict keys [{checkpoint_state_dict_key}] does not exist in the checkpoint."
            )
            try_default_key = True
    else:
        cleaned_checkpoint = checkpoint[checkpoint_state_dict_key]

    if try_default_key:
        if DEFAULT_STATE_DICT_KEY not in checkpoint:
            logger.warning(
                f"State dict keys [{DEFAULT_STATE_DICT_KEY}] does not exist in the checkpoint."
            )
        else:
            cleaned_checkpoint = checkpoint[DEFAULT_STATE_DICT_KEY]

    # remove some keys that might have been prepended due to training in ddp
    filtered_keys += ["_wrapped_model.", "_module", "module"]
    logger.info(f"Filtering Following keys from the checkppoint: {filtered_keys}")
    for key in filtered_keys:
        cleaned_checkpoint = filter_keys(cleaned_checkpoint, keys=[key])

    # custom filtered keys
    cleaned_checkpoint = filter_keys(cleaned_checkpoint, keys=filtered_keys)

    # now loda the checkpoint
    keys = model.load_state_dict(cleaned_checkpoint, strict=strict)
    if not strict:
        if keys.missing_keys:
            logger.warning(
                f"Found keys that are in the model state dict but not in the checkpoint: {keys.missing_keys}"
            )
        if keys.unexpected_keys:
            logger.warning(
                f"Found keys that are not in the model state dict but in the checkpoint: {keys.unexpected_keys}"
            )

    return model
