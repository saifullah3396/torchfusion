#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PYTHONPATH=$SCRIPT_DIR/../../../external/due_benchmark/:$PYTHONPATH

## declare the datasets, train_strategies and max_seq_lengths
declare -a datasets=(DocVQA PWC DeepForm TabFact WikiTableQuestions InfographicsVQA KleisterCharity)
declare -a train_strategies=(all_items concat all_items all_items concat all_items all_items)
declare -a max_seq_lengths=(1024 6144 6144 1024 4096 1024 6144)
declare -a ocr_engines=(microsoft_cv tesseract microsoft_cv tesseract microsoft_cv microsoft_cv microsoft_cv)

# get length of an array
arraylength=${#datasets[@]}

# use for loop to read all values and indexes
for (( i=0; i<${arraylength}; i++ ));
do
    # for docVQA only cv_microsoft has "common_format", all else fail
    echo "$i: creating dataset[${datasets[$i]}] with train_strategy[${train_strategies[$i]}], max_seq_length[${max_seq_lengths[$i]}] and ocr_engine[${ocr_engines[$i]}]"
    $SCRIPT_DIR/../../../scripts/analyze.sh -c prepare_datasets +run=prepare_due_benchmark_with_preprocess args/data_args=due_benchmark dataset_config_name=${datasets[$i]} max_encoder_length=${max_seq_lengths[$i]} train_strategy=${train_strategies[$i]} ocr_engine=${ocr_engines[$i]}
done