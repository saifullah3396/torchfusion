import importlib
from copy import deepcopy

from torchfusion.core.data.data_augmentations.general import DictTransform
from torchfusion.utilities.dataclasses.dacite_wrapper import from_dict
from torchfusion.utilities.module_import import ModuleLazyImporter


class TokenizerFactory:
    @staticmethod
    def create(
        name: str,
        kwargs: dict,
    ):
        tokenizer_class = ModuleLazyImporter.get_tokenizers().get(name, None)

        if tokenizer_class is not None:
            # lazy load class
            tokenizer_class = tokenizer_class()
        else:
            try:
                tokenizer_class = getattr(
                    importlib.import_module(".".join(name.split(".")[:-1])),
                    name.split(".")[-1],
                )
            except Exception as e:
                raise ValueError(
                    f"Tokenizer [{name}] is not supported. Available tokenizers: {ModuleLazyImporter.get_tokenizers()}"
                ) from e

        return from_dict(
            data_class=tokenizer_class,
            data=deepcopy(kwargs),
        )
