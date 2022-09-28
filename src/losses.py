import torch
from torch import nn, Tensor
from src.utils import nuclear_norm
import einops
from typing import Tuple

import sys

sys.path.append("..")


def get_objective(**kwargs):
    lmbda = kwargs["lmbda"]
    n_aug = kwargs["n_aug"]
    return MMCR_Loss(lmbda=lmbda, n_aug=n_aug)


class MMCR_Loss(nn.Module):
    def __init__(self, lmbda: float, n_aug: int):
        super(MMCR_Loss, self).__init__()
        self.lmbda = lmbda
        self.n_aug = n_aug

    def forward(self, z: Tensor) -> Tuple[Tensor, dict]:
        z_local = einops.rearrange(z, "(B N) C -> B C N", N=self.n_aug)

        batch_size = z_local.shape[0]

        centroids = torch.mean(z_local, dim=-1)
        local_nuc = nuclear_norm(z_local)
        global_nuc = nuclear_norm(centroids)

        loss = -1.0 * global_nuc + self.lmbda * local_nuc / batch_size

        loss_dict = {
            "loss": loss.item(),
            "local_nuc": local_nuc.item(),
            "global_nuc": global_nuc.item(),
        }

        return loss, loss_dict