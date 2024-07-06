import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

from pyparsing import abstractmethod
from torchfusion.core.models.utilities.checkpoints import setup_checkpoint
from torchfusion.core.utilities.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelConstructor:
    model_name: str = ""
    init_args: dict = field(default_factory=lambda: {})
    pretrained: bool = field(
        default=True,
        metadata={"help": ("Whether to load the model weights if available.")},
    )
    checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Checkpoint file name to load the model weights from."},
    )
    load_checkpoint_strict: bool = field(
        default=False, metadata={"help": "Whether to load the model weights strictly."}
    )
    checkpoint_state_dict_key: str = field(
        default="state_dict",
        metadata={"help": "The state dict key for checkpoint"},
    )
    checkpoint_filtered_keys: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The keys filtered from the checkpoint"},
    )
    # task of internal model can be different. This is task of torch model not of outer task model
    model_task: Optional[str] = field(
        default=None,
        metadata={"help": "Training task for which the model is loaded."},
    )
    cache_dir: str = field(
        default=os.environ.get("TORCH_FUSION_CACHE_DIR", "./cache/") + "/pretrained/",
        metadata={"help": "The location to store pretrained or cached models."},
    )

    def __post_init__(self):
        assert (
            self.model_name != ""
        ), "Model name must be provided for the model constructor."

    @abstractmethod
    def _init_model(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def __call__(
        self,
        checkpoint: Optional[str] = None,
        strict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        model = self._init_model(**kwargs)
        if checkpoint is None:
            checkpoint = self.checkpoint

        if strict is None:
            strict = self.load_checkpoint_strict

        if checkpoint is not None:
            setup_checkpoint(
                model,
                checkpoint,
                self.checkpoint_state_dict_key,
                strict=strict,
                filtered_keys=self.checkpoint_filtered_keys,
            )

        return model
