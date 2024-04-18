from __future__ import annotations

from typing import TYPE_CHECKING

import hydra
import ignite.distributed as idist
from hydra.core.hydra_config import HydraConfig

from torchfusion.core.training.fusion_trainer import FusionTrainer  # noqa

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(version_base=None, config_name="hydra")
def app(cfg: DictConfig) -> None:
    # get hydra config
    hydra_config = HydraConfig.get()

    # train and evaluate the model
    _, _ = FusionTrainer.run_train(cfg, hydra_config)

    # evaluate the model only on rank 0
    if idist.get_rank() == 0:
        _ = FusionTrainer.run_test(cfg, hydra_config)

    # wait for all processes to complete before exiting
    idist.barrier()


if __name__ == "__main__":
    app()
