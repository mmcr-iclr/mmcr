from __future__ import annotations
from PIL import Image, ImageOps, ImageFilter
import random
import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms.functional as TF
import torchvision
from torchvision import transforms
from PIL import Image

import sys
from turtle import down

sys.path.append("..")


# TODO: handle ablations as in CIFAR
class StlBatchTransform:
    def __init__(self, n_transform, train_transform=True, batch_transform=True):
        if train_transform is True:
            self.transform = transforms.Compose(
                [
                    transforms.RandomApply(
                        [
                            transforms.ColorJitter(
                                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                            )
                        ],
                        p=0.8,
                    ),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.RandomResizedCrop(
                        64,
                        scale=(0.2, 1.0),
                        ratio=(0.75, (4 / 3)),
                        interpolation=Image.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.43, 0.42, 0.39), (0.27, 0.26, 0.27)),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(70, interpolation=Image.BICUBIC),
                    transforms.CenterCrop(64),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.43, 0.42, 0.39), (0.27, 0.26, 0.27)),
                ]
            )
        self.n_transform = n_transform
        self.batch_transform = batch_transform

    def __call__(self, x):

        if self.batch_transform:
            C, H, W = TF.to_tensor(x).shape
            C_aug, H_aug, W_aug = self.transform(x).shape

            y = torch.zeros(self.n_transform, C_aug, H_aug, W_aug)
            for i in range(self.n_transform):
                y[i, :, :, :] = self.transform(x)
            return y
        else:
            return self.transform(x)


class CifarBatchTransform:
    def __init__(
        self,
        n_transform,
        train_transform=True,
        batch_transform=True,
        augmentation_indices=None,
        **kwargs,
    ):

        if train_transform is True:
            lst_of_transform = [
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
                ),
            ]

            if augmentation_indices is not None:
                for aug_ind in augmentation_indices:
                    lst_of_transform.pop(aug_ind)
                self.transform = transforms.Compose(lst_of_transform)

                print("List of transforms after augmentation ablation:")
                print(lst_of_transform)
            else:
                self.transform = transforms.Compose(lst_of_transform)
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
                    ),
                ]
            )
        self.n_transform = n_transform
        self.batch_transform = batch_transform

    def __call__(self, x):

        if self.batch_transform:
            C, H, W = TF.to_tensor(x).shape
            C_aug, H_aug, W_aug = self.transform(x).shape

            y = torch.zeros(self.n_transform, C_aug, H_aug, W_aug)
            for i in range(self.n_transform):
                y[i, :, :, :] = self.transform(x)
            return y
        else:
            return self.transform(x)


def get_datasets(
    dataset, n_aug, data_dir=None, **kwargs
):
    if data_dir is None:
        data_dir = "."
        download = True
    else:
        download = False
    if dataset == "stl10":
        train_data = torchvision.datasets.STL10(
            root=data_dir,
            split="train+unlabeled",
            transform=StlBatchTransform(
                train_transform=True, n_transform=n_aug),
            download=download,
        )
        memory_data = torchvision.datasets.STL10(
            root=data_dir,
            split="train",
            transform=StlBatchTransform(
                train_transform=False, batch_transform=False, n_transform=n_aug
            ),
            download=download,
        )
        test_data = torchvision.datasets.STL10(
            root=data_dir,
            split="test",
            transform=StlBatchTransform(
                train_transform=False, batch_transform=False, n_transform=n_aug
            ),
            download=download,
        )
    elif dataset == "cifar10":
        train_data = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=True,
            transform=CifarBatchTransform(
                train_transform=True, batch_transform=True, n_transform=n_aug, **kwargs
            ),
            download=download,
        )
        memory_data = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=True,
            transform=CifarBatchTransform(
                train_transform=False,
                batch_transform=False,
                n_transform=n_aug,
                **kwargs,
            ),
            download=download,
        )
        test_data = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=False,
            transform=CifarBatchTransform(
                train_transform=False,
                batch_transform=False,
                n_transform=n_aug,
                **kwargs,
            ),
            download=download,
        )
    elif dataset == "cifar100":
        train_data = torchvision.datasets.CIFAR100(
            root=data_dir,
            train=True,
            transform=CifarBatchTransform(
                train_transform=True, batch_transform=True, n_transform=n_aug, **kwargs
            ),
            download=download,
        )
        memory_data = torchvision.datasets.CIFAR100(
            root=data_dir,
            train=True,
            transform=CifarBatchTransform(
                train_transform=False,
                batch_transform=False,
                n_transform=n_aug,
                **kwargs,
            ),
            download=download,
        )
        test_data = torchvision.datasets.CIFAR100(
            root=data_dir,
            train=False,
            transform=CifarBatchTransform(
                train_transform=False,
                batch_transform=False,
                n_transform=n_aug,
                **kwargs,
            ),
            download=download,
        )

    return train_data, memory_data, test_data
