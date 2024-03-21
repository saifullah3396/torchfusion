from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelCheckpointArguments:
    # Model checkpoint directory name
    dir: str = "checkpoints"

    # How many checkpoints to save
    n_saved: int = 1

    # How many best checkpoints to save
    n_best_saved: int = 1

    # Which metric to monitor
    monitored_metric: Optional[str] = "val/loss"

    # Mode to use min/max
    mode: str = "min"

    # Checkpoints name prefix
    name_prefix: str = ""

    # Whether to save only the weights
    save_weights_only: bool = False

    # Whether to load only the weights
    load_weights_only: bool = False

    # If > 0, checkpoints are saved every n iteration steps
    every_n_steps: Optional[int] = None

    # If > 0, checkpoints are saved every n iteration epochs
    every_n_epochs: Optional[int] = None

    def __post_init__(self):
        if self.every_n_steps is not None and self.every_n_epochs is not None:
            raise RuntimeError(
                "model_checkpoint_config.every_n_steps and model_checkpoint_config.every_n_epochs are mutually exclusive"
            )

        if self.every_n_steps is None and self.every_n_epochs is None:
            self.every_n_epochs = 1

    @property
    def save_every_iters(self):
        if self.every_n_epochs is not None:
            return self.every_n_epochs
        else:
            return self.every_n_steps

    @property
    def save_per_epoch(self):
        if self.every_n_epochs is not None:
            return True
        else:
            return False
