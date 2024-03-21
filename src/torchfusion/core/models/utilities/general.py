import functools

import timm
import torch
from torch import nn


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def load_model_from_online_repository(
    model_name, num_labels=None, pretrained=True, use_timm=True, **kwargs
):
    if use_timm:
        if num_labels is None:
            model = timm.create_model(model_name, pretrained=pretrained, **kwargs)
        else:
            model = timm.create_model(
                model_name, pretrained=pretrained, num_classes=num_labels, **kwargs
            )
    else:
        model = torch.hub.load(
            "pytorch/vision:v0.10.0",
            model_name,
            pretrained=pretrained,
            verbose=False,
            **kwargs,
        )
    return model


def pad_sequences(sequences, padding_side, max_length, padding_elem):
    if padding_side == "right":
        if isinstance(sequences[0][0], list):
            padded_sequences = []
            for seq_list in sequences:
                padded_sequences.append([])
                for seq in seq_list:
                    padded_sequences[-1].append(
                        seq + [padding_elem] * (max_length - len(seq))
                    )
            return padded_sequences
        else:
            return [seq + [padding_elem] * (max_length - len(seq)) for seq in sequences]
    else:
        if isinstance(sequences[0][0], list):
            padded_sequences = []
            for seq_list in sequences:
                padded_sequences.append([])
                for seq in seq_list:
                    padded_sequences[-1].append(
                        [padding_elem] * (max_length - len(seq)) + seq
                    )
            return padded_sequences
        else:
            return [[padding_elem] * (max_length - len(seq)) + seq for seq in sequences]


def batch_norm_to_group_norm(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            num_channels = module.num_features

            # first level of current layer or model contains a batch norm --> replacing.
            def get_groups(num_channels, groups):
                if num_channels % groups != 0:
                    groups = groups // 2
                    groups = get_groups(num_channels, groups)
                return groups

            groups = get_groups(num_channels, 32)
            bn = rgetattr(model, name)
            gn = torch.nn.GroupNorm(groups, num_channels)
            rsetattr(model, name, gn)


def remove_lora_layers(model):
    from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear

    for name, module in model.named_modules():
        if isinstance(module, LoRACompatibleLinear):
            new_module = nn.Linear(module.in_features, module.out_features)
            new_module.__dict__.update(module.__dict__)
            rsetattr(model, name, new_module)
        if isinstance(module, LoRACompatibleConv):
            new_module = nn.Conv2d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
            )
            rsetattr(model, name, new_module)
