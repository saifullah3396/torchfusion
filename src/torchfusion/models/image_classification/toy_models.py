""" Base Model class for the any model from Timm Repository. """

from dataclasses import dataclass

import torch.nn.functional as F
from torch import nn

from torchfusion.core.models.args.fusion_model_config import FusionModelConfig
from torchfusion.core.models.classification.image import (
    FusionNNModelForImageClassification,
)


class ToyModelForCifar10Classification(FusionNNModelForImageClassification):
    @dataclass
    class Config(FusionNNModelForImageClassification.Config):
        pass

    def _build_classification_model(self):
        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1)
                self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)
                self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
                self.fc1 = nn.Linear(4 * 4 * 64, 500)
                self.dropout1 = nn.Dropout(0.5)
                self.fc2 = nn.Linear(500, 10)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.max_pool2d(x, 2, 2)
                x = F.relu(self.conv2(x))
                x = F.max_pool2d(x, 2, 2)
                x = F.relu(self.conv3(x))
                x = F.max_pool2d(x, 2, 2)
                x = x.view(-1, 4 * 4 * 64)
                x = F.relu(self.fc1(x))
                x = self.dropout1(x)
                x = self.fc2(x)
                return x

        return ToyModel()


class ToyModelForMNISTClassification(FusionNNModelForImageClassification):
    @dataclass
    class Config(FusionModelConfig):
        pass

    def _build_classification_model(self):
        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
                self.conv2 = nn.Conv2d(16, 32, 4, 2)
                self.fc1 = nn.Linear(32 * 4 * 4, 32)
                self.fc2 = nn.Linear(32, 10)

            def forward(self, x):
                # x of shape [B, 1, 28, 28]
                x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
                x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
                x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
                x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
                x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
                x = F.relu(self.fc1(x))  # -> [B, 32]
                x = self.fc2(x)  # -> [B, 10]
                return x

        return ToyModel()
