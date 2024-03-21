from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import hydra
from hydra.core.hydra_config import HydraConfig

from torchfusion.core.analyzer.analyzer import FusionAnalyzer
from torchfusion.core.analyzer.args import AnalyzerArguments
from torchfusion.core.args.args import FusionArguments

if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass
class FusionAnalyzerArguments(FusionArguments):
    """
    Add our our arguments to base arguments class here
    """

    analyzer_args: AnalyzerArguments = None


@hydra.main(version_base=None, config_name="hydra")
def app(cfg: DictConfig) -> None:
    # get hydra config
    hydra_config = HydraConfig.get()

    # evaluate the model
    _ = FusionAnalyzer.run(cfg, hydra_config, data_class=FusionAnalyzerArguments)


if __name__ == "__main__":
    app()
