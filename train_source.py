"""train_source.py: train the source model."""

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm
from pathlib import Path

import config
from utils import GrayscaleToRgb

if(config.dataset == 'FMNIST'):
    from fmnist_asset.models import Net
    model_path = Path('fmnist_asset/')
elif(config.dataset == 'SVHN'):
    from svhn_asset.models import Net
    model_path = Path('svhn_asset/')
elif(config.dataset == 'CIFAR10'):
    from cifar10_asset.models import Net
    model_path = Path('cifar10_asset/')
elif(config.dataset == 'CIFAR100'):
    from cifar100_asset.models import Net
    model_path = Path('cifar100_asset/')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_dataloaders(batch_size):

    if (config.dataset == 'FMNIST'):
        dataset = datasets.FashionMNIST(
            config.DATA_DIR / 'fmnist',
            train=True,
            download=True,
            transform=Compose(
                [
                    GrayscaleToRgb(),
                    ToTensor()]))
    if (config.dataset == 'SVHN'):
        dataset = datasets.SVHN(config.DATA_DIR / 'svhn',
                                split='train',
                                download=True,
                                transform=Compose([ToTensor()]))
    elif (config.dataset == 'CIFAR10'):
        dataset = datasets.CIFAR10(
            config.DATA_DIR / 'cifar10', train=True, download=True, transform=Compose([ToTensor()]))
    elif (config.dataset == 'CIFAR100'):
        dataset = datasets.CIFAR100(
            config.DATA_DIR / 'cifar100', train=True, download=True, transform=Compose([ToTensor()]))

    shuffled_indices = np.random.permutation(len(dataset))
    train_idx = shuffled_indices[:int(0.8 * len(dataset))]
    val_idx = shuffled_indices[int(0.8 * len(dataset)):]

    train_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True,
                              sampler=SubsetRandomSampler(train_idx),
                              num_workers=1, pin_memory=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, drop_last=False,
                            sampler=SubsetRandomSampler(val_idx),
                            num_workers=1, pin_memory=True)
    return train_loader, val_loader


def do_epoch(model, dataloader, criterion, optim=None):
    total_loss = 0
    total_accuracy = 0
    for x, y_true in tqdm(dataloader, leave=False):
        x, y_true = x.to(device), y_true.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y_true)

        if optim is not None:
            optim.zero_grad()
            loss.backward()
            optim.step()

        total_loss += loss.item()
        total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()
    mean_loss = total_loss / len(dataloader)
    mean_accuracy = total_accuracy / len(dataloader)

    return mean_loss, mean_accuracy


def main():
    batch_size = 64
    train_loader, val_loader = create_dataloaders(batch_size)

    model = Net().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, patience=10, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()

    best_accuracy = 0

    num_epochs = 100
    for epoch in range(1, num_epochs):
        model.train()
        train_loss, train_accuracy = do_epoch(
            model, train_loader, criterion, optim=optim)

        model.eval()
        with torch.no_grad():
            val_loss, val_accuracy = do_epoch(
                model, val_loader, criterion, optim=None)

        tqdm.write(
            f'EPOCH {epoch:03d}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f} '
            f'val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}')

        if val_accuracy > best_accuracy:
            print('Saving model...')
            best_accuracy = val_accuracy
            torch.save(
                model.state_dict(),
                model_path / 'trained_models/source.pt')

        lr_schedule.step(val_loss)


if __name__ == '__main__':
    main()
