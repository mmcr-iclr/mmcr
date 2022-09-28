import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
from torch import Tensor
from typing import Tuple


class Model(nn.Module):
    def __init__(
        self,
        feature_dim: int = 128,
        hidden_dim: int = 512,
        dataset: str = "cifar10",
    ):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == "conv1":
                module = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=1, bias=False
                )
            if dataset == "cifar10" or dataset == "cifar100":
                if not isinstance(module, nn.Linear) and not isinstance(
                    module, nn.MaxPool2d
                ):
                    self.f.append(module)
            elif dataset == "tiny_imagenet" or dataset == "stl10":
                if not isinstance(module, nn.Linear):
                    self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)

        # projection head
        self.g = nn.Sequential(
            nn.Linear(2048, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim, bias=False),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)

        return feature, out
