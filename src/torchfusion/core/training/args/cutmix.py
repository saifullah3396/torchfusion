from dataclasses import dataclass
from typing import Optional


@dataclass
class CutmixupArguments:
    mixup: float = 0
    cutmix: float = 0
    cutmix_minmax: Optional[float] = None
    mixup_prob: float = 1.0
    mixup_switch_prob: float = 0.5
    mixup_mode: str = "batch"

    def get_fn(self, num_classes, smoothing):
        from timm.data.mixup import Mixup

        if self.mixup > 0 or self.cutmix > 0.0 or self.cutmix_minmax is not None:
            return Mixup(
                mixup_alpha=self.mixup,
                cutmix_alpha=self.cutmix,
                cutmix_minmax=self.cutmix_minmax,
                prob=self.mixup_prob,
                switch_prob=self.mixup_switch_prob,
                mode=self.mixup_mode,
                label_smoothing=smoothing,
                num_classes=num_classes,
            )
        else:
            return None
