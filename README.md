## Environment Setup
Install the dependencies:
```
pip install -r requirements.txt
```

Setup environment variables:
```
export PYTHONPATH=<path/to/torchfusion>/src
export DATA_ROOT_DIR=/path/to/output/
export TORCH_FUSION_OUTPUT_DIR=/path/to/cache/

# can be any directory where datasets are cached and model training outputs are generated.
export TORCH_FUSION_OUTPUT_DIR=</your/output/dir>
```