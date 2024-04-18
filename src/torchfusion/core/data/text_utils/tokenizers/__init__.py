from torchfusion.core.utilities.module_import import ModuleLazyImporter

_import_structure = {
    "hf_tokenizer": [
        "HuggingfaceTokenizer",
    ],
    "tt_tokenizer": [
        "TorchTextTokenizer",
    ],
}
ModuleLazyImporter.register_tokenizers(__name__, _import_structure)
