from dataclasses import dataclass

from torchfusion.core.models.detection.base import FusionModelForObjectDetection
from torchfusion.core.utilities.logging import get_logger

logger = get_logger(__name__)


class FusionModelForImageObjectDetection(FusionModelForObjectDetection):
    @dataclass
    class Config(FusionModelForObjectDetection.Config):
        pass
