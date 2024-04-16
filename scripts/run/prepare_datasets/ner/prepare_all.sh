#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
$SCRIPT_DIR/../../../../scripts/analyze.sh -c prepare_datasets +run=ner/prepare_dataset dataset_config_name=default args/data_args=datasets/ner/cord visualize=false segment_level_layout=True
$SCRIPT_DIR/../../../../scripts/analyze.sh -c prepare_datasets +run=ner/prepare_dataset dataset_config_name=default args/data_args=datasets/ner/sroie visualize=false segment_level_layout=True
$SCRIPT_DIR/../../../../scripts/analyze.sh -c prepare_datasets +run=ner/prepare_dataset dataset_config_name=default args/data_args=datasets/ner/funsd visualize=false segment_level_layout=True
$SCRIPT_DIR/../../../../scripts/analyze.sh -c prepare_datasets +run=ner/prepare_dataset dataset_config_name=default args/data_args=datasets/ner/wild_receipts visualize=false segment_level_layout=True
$SCRIPT_DIR/../../../../scripts/analyze.sh -c prepare_datasets +run=ner/prepare_dataset dataset_config_name=kile args/data_args=datasets/ner/docile visualize=false segment_level_layout=True
