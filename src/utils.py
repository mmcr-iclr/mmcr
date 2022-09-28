import sched
import numpy as np
import random
import torch
from torch import Tensor
from torch import optim
import math

# I/O utilities
def gen_save_str(epoch, id=None):
    if id is None:
        id = str(random.randint(1000, 100000000000))

    return id + "_" + str(epoch) + "_.pt", id


def checkpoint_model(
    model,
    optimizer,
    args,
    id,
    epoch,
    base_folder,
    experiment_name,
    acc=None,
):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "args": args,
            "optim_state_dict": optimizer.state_dict(),
            "knn_accuracy": acc,
        },
        base_folder +
        + experiment_name
        + "/"
        + id
        + "_"
        + str(epoch)
        + "_.pt",
    )

# loss utilities
def nuclear_norm(F: Tensor) -> Tensor:
    U, S, Vt = torch.linalg.svd(F)

    return torch.sum(S)