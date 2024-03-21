from dataclasses import dataclass
from typing import Optional


@dataclass
class EarlyStoppingArguments:
    # Metric to monitor for early stopping
    monitored_metric: Optional[str] = "val/loss"

    # Minimum delta between subsequent steps
    min_delta: float = 0.0

    # Stopping patience
    patience: int = 3

    # Whether to accumulate delta
    cumulative_delta: bool = False

    # Mode to use, min/max
    mode: str = "min"
