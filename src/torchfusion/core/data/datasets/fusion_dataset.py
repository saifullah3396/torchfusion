from abc import ABC, abstractmethod

import datasets
from torchfusion.core.data.datasets.fusion_dataset_config import FusionDatasetConfig
from torchfusion.core.data.datasets.msgpack.builder import MsgpackBasedBuilder
from torchfusion.core.utilities.logging import get_logger


class FusionDataset(MsgpackBasedBuilder, ABC):
    BUILDER_CONFIGS = [
        FusionDatasetConfig(
            name="default",
            version=datasets.Version("1.0.0"),
            description="Default dataset config",
        ),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = get_logger(__name__)

    def _relative_data_dir(self, with_version=True, with_hash=False) -> str:
        return super()._relative_data_dir(
            with_version=with_version, with_hash=False
        )  # we do not add hash to keep it simple

    @abstractmethod
    def _dataset_features(self):
        raise NotImplementedError("Subclasses must implement _dataset_features()")

    def _info(self):
        return datasets.DatasetInfo(
            description=self.config.description,
            features=self._dataset_features(),
            supervised_keys=None,
            homepage=self.config.homepage,
            citation=self.config.citation,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = None
        if self.config.data_dir is None:
            if self.config.data_url is not None:
                data_dir = dl_manager.download_and_extract(self.config.data_url)
            else:
                raise ValueError("You must specify a data_url or a data_dir")
        else:
            data_dir = self.config.data_dir

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_dir": data_dir, "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"data_dir": data_dir, "split": "test"},
            ),
        ]

    def _generate_examples(self, *args, **kwargs):
        for key, sample in self._generate_examples_impl(*args, **kwargs):
            yield key, sample

    @abstractmethod
    def _generate_examples_impl(self, *args, **kwargs):
        raise NotImplementedError()
