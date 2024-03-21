#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PYTHONPATH=$PYTHONPATH:$SCRIPT_DIR/../src
export TORCH_FUSION_OUTPUT_DIR=$SCRIPT_DIR/../output
export TORCH_FUSION_CACHE_DIR=$SCRIPT_DIR/../cache

TYPE=standard
CONFIG_ROOT=""
POSITIONAL_ARGS=()

usage()
{
    echo "Usage:"
    echo "./analyze.sh -c/--config_root=<config_root>"
    echo ""
    echo " --config_root : Command to run. "
    echo " -h | --help : Displays the help"
    echo ""
}

while [[ $# -gt 0 ]]; do
    case $1 in
    -h|--help)
        shift # past argument
        usage
        exit
        ;;
    -c|--config_root)
        CONFIG_ROOT="$2"
        shift # past argument
        shift # past value
        ;;
    *)
        POSITIONAL_ARGS+=("$1") # save positional arg
        shift # past argument
        ;;
    esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters
python3 $SCRIPT_DIR/../src/torchfusion/core/analyzer/analyze.py --config-path ../../../../cfg/$CONFIG_ROOT "${@:1}"
