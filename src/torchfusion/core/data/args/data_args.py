import os
from dataclasses import dataclass, field
from typing import List, Optional, Union

from torchfusion.core.args.args_base import ClassInitializerArgs
from torchfusion.core.data.args.data_loader_args import DataLoaderArguments


@dataclass
class DataArguments:
    # Name of the dataset
    dataset_name: str = ""

    # Dataset directory path
    dataset_dir: Optional[str] = None

    # Dataset config name
    dataset_config_name: Optional[Union[str, List[str]]] = field(default="default")

    # Dataset cacher arguments
    dataset_cache_dir: str = field(
        default=os.environ.get("TORCH_FUSION_CACHE_DIR", "./cache/") + "/huggingface",
        metadata={"help": ("Directory to cache the dataset.")},
    )

    # use features cached in data directory instead of actual dataset
    features_path: Optional[str] = field(default=None)

    # Name of the cache file
    cache_file_name: str = field(default="default")

    # Whether to use caching
    enable_caching: bool = field(default=True)

    # whether to compute fid stats
    compute_dataset_statistics: bool = field(default=False)

    # filename for stats
    stats_filename: str = field(default="stats")

    # number of samples to use for stats
    dataset_statistics_n_samples: int = 5000

    # Number of processes to use for processing data for cache
    num_proc: int = field(default=4)

    # Whether to stream dataset (for hf)
    streaming: bool = field(default=False)

    # Whether dataset auth token is required
    use_auth_token: bool = field(default=False)

    # Train validation sampling arguments
    train_val_sampler: Optional[ClassInitializerArgs] = field(default=None)

    # Arguments related to defining default data augmentations for training.
    train_preprocess_augs: Optional[
        Union[ClassInitializerArgs, List[ClassInitializerArgs]]
    ] = field(default=None)

    # Arguments related to defining default data augmentations for evaluation.
    eval_preprocess_augs: Optional[
        Union[ClassInitializerArgs, List[ClassInitializerArgs]]
    ] = field(default=None)

    # Arguments related to defining default data augmentations for training.
    train_realtime_augs: Optional[
        Union[ClassInitializerArgs, List[ClassInitializerArgs]]
    ] = field(default=None)

    # Arguments related to defining default data augmentations for training.
    eval_realtime_augs: Optional[
        Union[ClassInitializerArgs, List[ClassInitializerArgs]]
    ] = field(default=None)

    # Arguments related to data loading or specifically torch dataloaders.
    data_loader_args: DataLoaderArguments = field(
        default_factory=lambda: DataLoaderArguments(),
    )

    # Preprocess batch size
    preprocess_batch_size: int = field(default=1000)

    # Any additional argument required specifically for the dataset.
    dataset_kwargs: Union[dict, List[dict]] = field(
        default_factory=lambda: {},
        metadata={
            "help": ("Any additional argument required specifically for the dataset.")
        },
    )

    def __post_init__(self):
        if self.train_preprocess_augs is not None and not isinstance(
            self.train_preprocess_augs, list
        ):
            self.train_preprocess_augs = [self.train_preprocess_augs]
        if self.eval_preprocess_augs is not None and not isinstance(
            self.eval_preprocess_augs, list
        ):
            self.eval_preprocess_augs = [self.eval_preprocess_augs]
        if self.train_realtime_augs is not None and not isinstance(
            self.train_realtime_augs, list
        ):
            self.train_realtime_augs = [self.train_realtime_augs]
        if self.eval_realtime_augs is not None and not isinstance(
            self.eval_realtime_augs, list
        ):
            self.eval_realtime_augs = [self.eval_realtime_augs]

        if isinstance(self.dataset_config_name, list):
            assert isinstance(self.dataset_kwargs, list)
            assert len(self.dataset_config_name) == len(self.dataset_kwargs)
