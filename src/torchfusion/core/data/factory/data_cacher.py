from __future__ import annotations

from torchfusion.core.utilities.module_import import ModuleLazyImporter


class DataCacherFactory:
    @staticmethod
    def create(strategy: str, **kwargs):
        data_cacher_class = ModuleLazyImporter.get_data_cachers().get(strategy, None)
        if data_cacher_class is None:
            raise ValueError(f"DataCacher [{strategy}] is not supported.")
        data_cacher_class = data_cacher_class()
        return data_cacher_class(**kwargs)
