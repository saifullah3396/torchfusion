from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Union

from torchfusion.core.training.args.cutmix import CutmixupArguments
from torchfusion.core.training.args.early_stopping import EarlyStoppingArguments
from torchfusion.core.training.args.ema import ModelEmaArguments
from torchfusion.core.training.args.model_checkpoint import ModelCheckpointArguments
from torchfusion.core.training.optim.args import OptimizerArguments
from torchfusion.core.training.sch.args import LRSchedulerArguments, WDSchedulerArguments


@dataclass
class TrainingArguments:
    """
    Arguments related to the training loop.
    """

    cls_name = "training_args"

    # Experiment name
    experiment_name: str = "default_exp"

    # Number of updates steps to accumulate before performing a backward/update pass
    gradient_accumulation_steps: int = 1

    # Whether to use gradient clipping or not
    enable_grad_clipping: bool = False

    # Max gradient norm
    max_grad_norm: float = 1.0

    # Min number of training epochs to perform
    min_epochs: Optional[float] = None

    # Max number of training epochs to perform
    max_epochs: Optional[float] = 1

    # Linear warmup over warmup_ratio fraction of total steps
    warmup_ratio: Optional[float] = None

    # Linear warmup over warmup_steps
    warmup_steps: Optional[int] = None

    # N update steps to log on
    logging_steps: int = 10

    # Whether to use AMP for training
    with_amp: bool = False

    # Whether to use AMP for training
    with_amp_inference: bool = False

    # Whether to stop training when getting NaN values
    stop_on_nan: bool = True

    # Whether to push data to GPU in non-blocking or blocking manner.
    non_blocking_tensor_conv: bool = False

    # How many times to run evaluation
    eval_every_n_epochs: Optional[Union[int, float]] = 1

    # whether to run eval on startup
    eval_on_start: bool = False

    # How many times to run evaluation
    visualize_every_n_epochs: Optional[Union[int, float]] = 1

    # whether to run eval on startup
    visualize_on_start: bool = False
    
    # Minimum training epochs for eval
    min_train_epochs_for_best: Optional[int] = 1

    # Whether to evaluate the metrics for training loop
    eval_training: Optional[bool] = True

    # Whether to clear cuda cache before each training_epoch
    clear_cuda_cache: Optional[bool] = True

    # Whether to log gpu stats
    log_gpu_stats: Optional[bool] = False

    # Whether to log outputs to tensorboard
    log_to_tb: Optional[bool] = True

    # Whether to load best or last checkpoints whether for resume
    load_best_checkpoint_resume: Optional[bool] = False

    # Whether to resume training from checkpoint if available
    resume_from_checkpoint: Optional[bool] = True

    # Checkpoint resume file name
    resume_checkpoint_file: Optional[str] = None

    # Test checkpoint file name
    test_checkpoint_file: Optional[str] = None

    # Whether to enable checkpoint at all
    enable_checkpointing: bool = True

    # Whether to convert batchnorm layers to syncnorm for ddp case
    sync_batchnorm: bool = field(
        default=True,
        metadata={"help": "Whether to synchronize batches accross multiple GPUs."},
    )
    # Model checkpoint related arguments
    model_checkpoint_config: ModelCheckpointArguments = field(
        default_factory=lambda: ModelCheckpointArguments(),
        metadata={"help": "ModelCheckpoint configuration that defines the saving strategy " "of model checkpoints."},
    )

    # Early stopping config
    early_stopping_args: EarlyStoppingArguments = field(default_factory=lambda: EarlyStoppingArguments())

    # Label smooth amount. Applies if > 0
    smoothing: float = 0.0

    # Cutmix and mixup arguments
    cutmixup_args: CutmixupArguments = field(
        default_factory=lambda: CutmixupArguments(0),
    )

    # EMA related arguments if required
    model_ema_args: ModelEmaArguments = field(default_factory=lambda: ModelEmaArguments())

    # Whether to use ema for validation
    use_ema_for_val: bool = False

    # output metric transforms
    outputs_to_metric: Optional[List[str]] = field(default_factory=lambda: ["loss"])

    # Optimizer configurations
    optimizers: Mapping[str, OptimizerArguments] = field(  # no support in omegaconf?
        # optimizers: Any = field(
        default_factory=lambda: {"default": OptimizerArguments()},
    )

    # Learning rate scheduler config
    lr_schedulers: Optional[Mapping[str, LRSchedulerArguments]] = field(
        default_factory=lambda: {"default": LRSchedulerArguments()},
        metadata={"help": "Learning rate scheduler."},
    )

    # Weight decay scheduler config
    wd_schedulers: Optional[Mapping[str, WDSchedulerArguments]] = field(
        default_factory=lambda: {"default": WDSchedulerArguments()},
        metadata={"help": "Weight decay scheduler."},
    )

    # test run
    test_run: bool = field(
        default=False,
        metadata={"help": "Set this to true for a single step run of training."},
    )

    def __post_init__(self):
        import logging

        if self.warmup_ratio is not None:
            if self.warmup_ratio < 0 or self.warmup_ratio > 1:
                raise ValueError("warmup_ratio must lie in range [0,1]")
            elif self.warmup_ratio is not None and self.warmup_steps is not None:
                logging.info(
                    "Both warmup_ratio and warmup_steps given, warmup_steps will override"
                    " any effect of warmup_ratio during training"
                )

        if self.warmup_ratio is None:
            self.warmup_ratio = 0
        if self.warmup_steps is None:
            self.warmup_steps = 0

        # if eval_every_n_epochs is a float maybe its less than an epoch
        if self.eval_every_n_epochs > 1.0:
            self.eval_every_n_epochs = int(self.eval_every_n_epochs)
