import torch
import torch.nn as nn
import torch.optim as optim

# from thop import profile, clever_format
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

import src.utils as utils
from src.data import (
    get_datasets,
    CifarBatchTransform,
    StlBatchTransform,
)

import torchvision

###  KNN based evaluation, for use during self-supervised pretraining to track progress ###
def test_one_epoch(
    net,
    memory_data_loader,
    test_data_loader,
    c,
    temperature=0.5,
    k=200,
):
    net.eval()
    total_top1, total_top5, total_num, feature_bank, target_bank = 0.0, 0.0, 0, [], []
    with torch.no_grad():
        # generate feature bank and target bank
        for data_tuple in memory_data_loader:
            data, target = data_tuple
            target_bank.append(target)
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = (
            torch.cat(target_bank, dim=0).contiguous().to(feature_bank.device)
        )
        # loop test data to predict the label by weighted knn search
        for data_tuple in test_data_loader:
            data, target = data_tuple
            data, target = data.cuda(
                non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(
                feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices
            )
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(
                data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(
                dim=-1, index=sim_labels.view(-1, 1), value=1.0
            )
            # weighted score ---> [B, C]
            pred_scores = torch.sum(
                one_hot_label.view(data.size(0), -1, c) *
                sim_weight.unsqueeze(dim=-1),
                dim=1,
            )

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum(
                (pred_labels[:, :1] == target.unsqueeze(
                    dim=-1)).any(dim=-1).float()
            ).item()
            total_top5 += torch.sum(
                (pred_labels[:, :5] == target.unsqueeze(
                    dim=-1)).any(dim=-1).float()
            ).item()

    return total_top1 / total_num * 100, total_top5 / total_num * 100


### Linear classifier, for evaluating after unsupervised training has been completed ###
class Net(nn.Module):
    def __init__(self, num_class, pretrained_path, dataset):
        super(Net, self).__init__()

        # encoder
        from src.models import Model

        self.f = Model(dataset=dataset).f
        # classifier
        self.fc = nn.Linear(2048, num_class, bias=True)
        pretrained = torch.load(pretrained_path, map_location="cpu")
        state_dict = pretrained["state_dict"]
        self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out


# train or test linear classifier for one epoch
def train_val(net, data_loader, train_optimizer, epoch):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = (
        0.0,
        0.0,
        0.0,
        0,
        tqdm(data_loader),
    )
    with (torch.enable_grad() if is_train else torch.no_grad()):
        loss_criterion = nn.CrossEntropyLoss()
        for data, target in data_bar:
            data, target = data.cuda(
                non_blocking=True), target.cuda(non_blocking=True)
            out = net(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum(
                (prediction[:, 0:1] == target.unsqueeze(
                    dim=-1)).any(dim=-1).float()
            ).item()
            total_correct_5 += torch.sum(
                (prediction[:, 0:5] == target.unsqueeze(
                    dim=-1)).any(dim=-1).float()
            ).item()

            data_bar.set_description(
                "{} Epoch: [{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%".format(
                    "Train" if is_train else "Test",
                    epoch,
                    total_loss / total_num,
                    total_correct_1 / total_num * 100,
                    total_correct_5 / total_num * 100,
                )
            )

    return (
        total_loss / total_num,
        total_correct_1 / total_num * 100,
        total_correct_5 / total_num * 100,
    )


def train_classifier(
    dataset: str,
    model_path: str,
    data_dir: str = None,
    batch_size: int = 1024,
    epochs: int = 50,
    lr: float = 1e-2,
):

    if data_dir is None:
        data_dir = "."
        download = True
    else:
        download = False
    if dataset == "cifar10":
        train_data = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=True,
            transform=CifarBatchTransform(
                train_transform=True, batch_transform=False, n_transform=1
            ),
            download=download,
        )
        test_data = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=False,
            transform=CifarBatchTransform(
                train_transform=False, batch_transform=False, n_transform=1
            ),
            download=download,
        )
    if dataset == "cifar100":
        train_data = torchvision.datasets.CIFAR100(
            root=data_dir,
            train=True,
            transform=CifarBatchTransform(
                train_transform=True, batch_transform=False, n_transform=1
            ),
            download=download,
        )
        test_data = torchvision.datasets.CIFAR100(
            root=data_dir,
            train=False,
            transform=CifarBatchTransform(
                train_transform=False, batch_transform=False, n_transform=1
            ),
            download=download,
        )
    elif dataset == "stl10":
        train_data = torchvision.datasets.STL10(
            root=data_dir,
            split="train",
            transform=StlBatchTransform(
                train_transform=True, batch_transform=False, n_transform=1
            ),
            download=download,
        )
        test_data = torchvision.datasets.STL10(
            root=data_dir,
            split="test",
            transform=StlBatchTransform(
                train_transform=False, batch_transform=False, n_transform=1
            ),
            download=download,
        )

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True
    )

    model = Net(
        num_class=len(train_data.classes), pretrained_path=model_path, dataset=dataset
    ).cuda()

    optimizer = optim.Adam(model.fc.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        # train one epoch
        train_loss, train_acc_1, train_acc_5 = train_val(
            model, train_loader, optimizer, epoch
        )
        # test one epoch
        test_loss, test_acc_1, test_acc_5 = train_val(
            model, test_loader, None, epoch)

        scheduler.step()

    return None

if __name__ == "__main__":
    pretrained_path = "/path/to/pretrained.pt" # actual path must be provided
    train_classifier(
        dataset_name="cifar10",
        model_path=pretrained_path,
        data_dir=None,
        batch_size=1024,
        epochs=50,
        lr=0.01
    )