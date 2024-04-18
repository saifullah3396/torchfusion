#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export PYTHONPATH=$SCRIPT_DIR/../src:$SCRIPT_DIR/../external/opacus

TYPE=standard
CFG_ROOT="main"
POSITIONAL_ARGS=()

usage() {
  echo "Usage:"
  echo "./train.sh --type=<type>"
  echo ""
  echo " --type : Command to run. "
  echo " -h | --help : Displays the help"
  echo ""
}

while [[ $# -gt 0 ]]; do
  case $1 in
  -h | --help)
    shift # past argument
    usage
    exit
    ;;
  -t | --type)
    TYPE="$2"
    shift # past argument
    shift # past value
    ;;
  -c | --cfg_root)
    CFG_ROOT="$2"
    shift # past argument
    shift # past value
    ;;
  *)
    POSITIONAL_ARGS+=("$1") # save positional arg
    shift                   # past argument
    ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters
python3 $SCRIPT_DIR/../src/torchfusion/runners/prepare_dataset.py --config-path ../../../cfg/$CFG_ROOT --config-name prepare_dataset "${@:1}"
