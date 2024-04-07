#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
$SCRIPT_DIR/../../../../scripts/analyze.sh -c prepare_datasets +run=layout/prepare_text_dataset dataset_config_name=default args/data_args=layout/docbank
