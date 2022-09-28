import random
from pathlib import Path
from tqdm import tqdm
import einops
from src.test import test_one_epoch as test
from src.losses import get_objective
from src.data import get_datasets
from src.models import Model
from src.utils import *
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch import optim
import torch.nn as nn
import os
import sys

sys.path.append("..")


# train for one epoch
def train_one_epoch(
    net,
    data_loader,
    train_optimizer,
    objective,
    epoch,
    epochs,
    batch_size: int,
):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    c = len(data_loader) * (epoch - 1)
    for step, data_tuple in enumerate(train_bar, start=c):
        img_batch, labels = data_tuple
        img_batch = einops.rearrange(img_batch, "B N C H W -> (B N) C H W")

        features, out = net(img_batch.cuda(non_blocking=True))
        loss, loss_dict = objective(out)

        train_optimizer.zero_grad()
        features, out = net(img_batch.cuda(non_blocking=True))
        loss, loss_dict = objective(out)
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description(
            "Train Epoch: [{}/{}] Loss: {:.4f} ".format(
                epoch,
                epochs,
                total_loss / total_num,
            )
        )

        c += 1

    return total_loss / total_num


def train(
    dataset_name: str,
    n_aug: int,
    batch_size: int,
    epochs: int,
    lr: float,
    objective: str,
    lmbda: float,
    tau: float = 0.5,
    feature_dim: int = 128,
    results_file: str = "results",
    experiment_name: str = "",
):

    args = locals()
    # data prepare
    train_data, memory_data, test_data = get_datasets(dataset_name, n_aug)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
    )
    memory_loader = DataLoader(
        memory_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True
    )

    # model setup and optimizer config
    model = Model(feature_dim=feature_dim, dataset=dataset_name).cuda()
    save_str, id = gen_save_str(epoch=0)

    c = len(memory_data.classes)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    # save model at intialization
    checkpoint_model(
        model=model,
        optimizer=optimizer,
        args=args,
        id=id,
        epoch=0,
        acc=-1,
        experiment_name=experiment_name,
    )

    if not os.path.exists(results_file):
        os.mkdir(results_file)

    best_acc = 0.0
    for epoch in range(0, epochs):
        train_loss = train_one_epoch(
            net=model,
            data_loader=train_loader,
            otpimizer=optimizer,
            objective=get_objective(
                objective,
                lmbda=lmbda,
                n_aug=n_aug,
                tau=tau,
            ),
            batch_size=batch_size,
            epoch=epoch,
            epochs=epochs,
            lr=lr,
        )
        if epoch % 1 == 0:
            # track progress with knn
            test_acc_1, test_acc_5 = test(
                model, memory_loader, test_loader, c, epoch
            )
            if test_acc_1 > best_acc:
                best_acc = test_acc_1
                checkpoint_model(
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    id=id,
                    epoch="top",
                    acc=test_acc_1,
                    experiment_name=experiment_name,
                )
        if epoch % 5 == 0:
            checkpoint_model(
                model=model,
                optimizer=optimizer,
                args=args,
                id=id,
                epoch=epoch + 1,
                acc=test_acc_1,
                experiment_name=experiment_name,
            )

    return None

if __name__ == "__main__":
    train(
        dataset_name="cifar10",
        n_aug=40,
        batch_size=32,
        epochs=500,
        lr=1e-3,
        lmbda=0.0,
        results_file="results",
        experiment_name="default"
    )
