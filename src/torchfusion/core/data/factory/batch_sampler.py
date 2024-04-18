from __future__ import annotations

from torchfusion.core.utilities.module_import import ModuleLazyImporter


def batch_sampler_wrapper(batch_sampler_class, **kwargs):
    def wrap(sampler):
        return batch_sampler_class(sampler=sampler, **kwargs)

    return wrap


class BatchSamplerFactory:
    @staticmethod
    def create(strategy: str, **kwargs):
        if strategy == "":
            return

        batch_sampler_class = ModuleLazyImporter.get_batch_samplers().get(
            strategy, None
        )
        if batch_sampler_class is None:
            raise ValueError(f"BatchSampler [{strategy}] is not supported.")
        batch_sampler_class = batch_sampler_class()
        return batch_sampler_wrapper(batch_sampler_class=batch_sampler_class, **kwargs)
