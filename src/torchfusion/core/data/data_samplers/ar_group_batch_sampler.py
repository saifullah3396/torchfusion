"""
Defines the GroupBatchSampler batch sampling strategy.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from torchfusion.core.data.data_samplers.group_batch_sampler import GroupBatchSampler

if TYPE_CHECKING:
    from torch.utils.data.sampler import Sampler


class AspectRatioGroupBatchSampler(GroupBatchSampler):
    """
    Groups the input sample images based on their aspect ratio.
    """

    def __init__(self, sampler: Sampler, group_factor: int, batch_size: int):
        from torchfusion.core.data.data_samplers.utilities import (
            create_aspect_ratio_groups,
        )

        group_ids = create_aspect_ratio_groups(
            sampler.data_source,
            k=group_factor,
        )

        super().__init__(sampler=sampler, group_ids=group_ids, batch_size=batch_size)
