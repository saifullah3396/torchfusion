from __future__ import annotations

import dataclasses
import re
import textwrap
from pathlib import Path
from typing import Generator, Union

import torch
import yaml


def indent_string(s: str, ind):
    return textwrap.indent(s, ind)


def str_to_underscored_lower(s: str):
    return "_".join(l.lower() for l in re.findall("[A-Z][^A-Z]*", s))


def check_max_len(x: list, max_len: int = 2):
    if len(x) != 2:
        raise ValueError("List should of size [{max_len}]")


def make_dir(path: Union[str, Path]):
    if not path.exists():
        path.mkdir(parents=True)


def concatenate_list_dict_to_dict(list_dict):
    output = {}
    for d in list_dict:
        for k, v in d.items():
            if k not in output:
                output[k] = []
            output[k].append(v)
    output = {k: torch.cat(v) if len(v[0].shape) > 0 else torch.tensor(v) for k, v in output.items()}
    return output


def drange(
    min_val: Union[int, float], max_val: Union[int, float], step_val: Union[int, float]
) -> Generator[Union[int, float], None, None]:
    curr = min_val
    while curr < max_val:
        yield curr
        curr += step_val


def generate_default_config(output: str):
    from torchfusion.core.args.args import FusionArguments

    args = FusionArguments()
    d = dataclasses.asdict(args)
    with open(output, "w") as f:
        yaml.dump(d, f)
