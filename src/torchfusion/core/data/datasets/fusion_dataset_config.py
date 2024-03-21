import dataclasses
import typing

import datasets
from datasets.features import Features

logger = datasets.logging.get_logger(__name__)


@dataclasses.dataclass
class FusionDatasetConfig(datasets.BuilderConfig):
    """BuilderConfig for FusionDataset"""

    data_url: str = None
    homepage: str = None
    citation: str = None
    license: str = None
    preprocess_transforms: typing.Any = None
    cache_file_name: str = None

    def create_config_id(
        self,
        config_kwargs: dict,
        custom_features: typing.Optional[Features] = None,
    ) -> str:
        # hashing causes all sorts of rinse and repeat of dataset caching again and again for even smallest of changes
        # so we don't use it here and keep it simple

        # if self.cache_file_name is not None:
        #     return super().create_config_id(config_kwargs, custom_features) + "-" + self.cache_file_name
        # else:
        #     return super().create_config_id(config_kwargs, custom_features)

        if self.cache_file_name is not None:
            config_id = self.name + "-" + self.cache_file_name
            return config_id
        else:
            return self.name
