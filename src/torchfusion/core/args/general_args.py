import os
from dataclasses import dataclass, field
from typing import Optional

from torchfusion.core.args.args_base import ArgumentsBase


@dataclass
class GeneralArguments(ArgumentsBase):
    """
    General initialization arguments.
    """

    root_output_dir: str = field(
        default=f"{os.environ.get('TORCH_FUSION_OUTPUT_DIR', './output')}",
        metadata={"help": ("Directory where the training output is stored.")},
    )
    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    do_test: bool = field(
        default=True, metadata={"help": "Whether to run evaluation on the test set."}
    )
    do_val: bool = field(
        default=True,
        metadata={"help": "Whether to run evaluation on the validation set."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed that will be set at the beginning of training."},
    )
    deterministic: bool = field(
        default=False,
        metadata={"help": "Whether to make the training process deterministic."},
    )
    backend: Optional[str] = field(
        default="nccl",
        metadata={"help": "Distributed backend to use [gloo/nccl/horovard]."},
    )
    n_devices: int = field(
        default=1,
        metadata={"help": "The number of gpus to use for training."},
    )
