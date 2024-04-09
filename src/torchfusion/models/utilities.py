from typing import List

from torch import nn


def find_layer_in_model(model: nn.Module, layer_name: str):
    # get encoder
    layer = [x for x, m in model.named_modules() if x == layer_name]
    if len(layer) == 0:
        raise ValueError(f"Encoder layer {layer_name} not found in the model.")
    return layer[0]


def freeze_layers_by_name(model: nn.Module, layer_names: List[str]):
    for layer_name in layer_names:
        layer = find_layer_in_model(model, layer_name)
        for p in layer.parameters():
            p.requires_grad = False


def freeze_layers(layers):
    for layer in layers:
        for p in layer.parameters():
            p.requires_grad = False
