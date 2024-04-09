#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
$SCRIPT_DIR/../../../../scripts/analyze.sh -c prepare_datasets +run=ner/prepare_dataset dataset_config_name=default args/data_args=ner/cord visualize=false segment_level_layout=True
#$SCRIPT_DIR/../../../../scripts/analyze.sh -c prepare_datasets +run=ner/prepare_dataset dataset_config_name=default args/data_args=ner/sroie visualize=false segment_level_layout=True
#$SCRIPT_DIR/../../../../scripts/analyze.sh -c prepare_datasets +run=ner/prepare_dataset dataset_config_name=default args/data_args=ner/funsd visualize=false segment_level_layout=True
#$SCRIPT_DIR/../../../../scripts/analyze.sh -c prepare_datasets +run=ner/prepare_dataset dataset_config_name=default args/data_args=ner/wild_receipts visualize=false segment_level_layout=True
#$SCRIPT_DIR/../../../../scripts/analyze.sh -c prepare_datasets +run=ner/prepare_dataset dataset_config_name=kile args/data_args=ner/docile visualize=false segment_level_layout=True
