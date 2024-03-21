from dataclasses import dataclass, field
from typing import Optional

from torchfusion.core.args.args_base import ClassInitializerArgs


@dataclass
class DataLoaderArguments:
    # Training batch size
    per_device_train_batch_size: int = 64

    # Evaluation batch size
    per_device_eval_batch_size: int = 64

    # Whether to drop last batch in data
    dataloader_drop_last: bool = False

    # Whether to shuffle the data
    shuffle_data: bool = True

    # Whether to pin memory for data loading
    pin_memory: bool = True

    # Dataloader number of workers
    dataloader_num_workers: int = 4

    # Maximum training samples to use
    max_train_samples: Optional[int] = None

    # Maximum val samples to use
    max_val_samples: Optional[int] = None

    # Maximum test samples to use
    max_test_samples: Optional[int] = None

    # The batch sampler arguments for using custom batch samplers for training
    train_batch_sampler: ClassInitializerArgs = field(
        default_factory=lambda: ClassInitializerArgs()
    )

    # The batch sampler arguments for using custom batch samplers for evaluation
    eval_batch_sampler: ClassInitializerArgs = field(
        default_factory=lambda: ClassInitializerArgs()
    )

    # Whether to replace test set for validation set
    use_test_set_for_val: bool = False

    # Whether to replace validation set for test set
    use_val_set_for_test: bool = False
