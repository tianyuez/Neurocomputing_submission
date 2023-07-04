"""train_at_ensemble.py: EAT."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor

from pathlib import Path
from tqdm import tqdm
import torchattacks

import config
from utils import GrayscaleToRgb
from test_util import test_util

if(config.dataset == 'FMNIST'):
    from fmnist_asset.models import Net
    from fmnist_asset.models_ensemble import Net1, Net2
    model_path = Path('fmnist_asset/')
elif(config.dataset == 'SVHN'):
    from svhn_asset.models import Net
    from svhn_asset.models_ensemble import Net1, Net2
    model_path = Path('svhn_asset/')
elif(config.dataset == 'CIFAR10'):
    from cifar10_asset.models import Net
    from cifar10_asset.models_ensemble import Net1, Net2
    model_path = Path('cifar10_asset/')
elif(config.dataset == 'CIFAR100'):
    from cifar100_asset.models import Net
    from cifar100_asset.models_ensemble import Net1, Net2
    model_path = Path('cifar100_asset/')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter()

    model = Net().to(device)
    model.load_state_dict(
        torch.load(
            model_path / 'trained_models/source.pt'))

    model1 = Net1().to(device)
    model1.load_state_dict(
        torch.load(
            model_path / 'trained_models/ensemble1.pt'))

    model2 = Net2().to(device)
    model2.load_state_dict(
        torch.load(
            model_path / 'trained_models/ensemble2.pt'))

    if (config.dataset == 'FMNIST'):
        dataset_training = datasets.FashionMNIST(
            config.DATA_DIR / 'fmnist',
            train=True,
            download=True,
            transform=Compose(
                [
                    GrayscaleToRgb(),
                    ToTensor()]))
    if (config.dataset == 'SVHN'):
        dataset_training = datasets.SVHN(
            config.DATA_DIR / 'svhn', split='train', download=True, transform=Compose([ToTensor()]))
    elif (config.dataset == 'CIFAR10'):
        dataset_training = datasets.CIFAR10(
            config.DATA_DIR / 'cifar10', train=True, download=True, transform=Compose([ToTensor()]))
    elif (config.dataset == 'CIFAR100'):
        dataset_training = datasets.CIFAR100(
            config.DATA_DIR / 'cifar100', train=True, download=True, transform=Compose([ToTensor()]))

    loader_source = DataLoader(
        dataset_training,
        batch_size=256,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True)
    loader_at = DataLoader(
        dataset_training,
        batch_size=256,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True)

    optim_at = torch.optim.Adam(model.parameters())

    accuracy_so_far = 0

    for epoch in range(1, config.num_epochs):
        batches = zip(loader_source, loader_at)
        n_batches = min(len(loader_source), len(loader_at))

        for (x_source, labels_source), (x_at, labels_at) in tqdm(
                batches, leave=False, total=n_batches):

            attack_at = torchattacks.FGSM(model, eps=config.eps)
            attack_ens1 = torchattacks.FGSM(model1, eps=config.eps)
            attack_ens2 = torchattacks.FGSM(model2, eps=config.eps)

            x_at = attack_at(x_at, labels_at)
            x_ens1 = attack_ens1(x_at, labels_at)
            x_ens2 = attack_ens2(x_at, labels_at)

            x = torch.cat([x_source.cpu(), x_at.cpu(),
                           x_ens1.cpu(), x_ens2.cpu()])
            x = x.to(device)
            labels_at_pred = model(x)

            labels_at = torch.cat(
                [labels_source, labels_at, labels_at, labels_at])
            labels_at = labels_at.to(device)
            loss_at = F.cross_entropy(labels_at_pred, labels_at.to(device))
            optim_at.zero_grad()
            loss_at.backward()
            optim_at.step()

        test_accuracy = test_util(
            model,
            model,
            device,
            epoch,
            writer,
            'EAT',
            'PGD20')

        if test_accuracy > accuracy_so_far:
            torch.save(
                model.state_dict(),
                model_path / 'trained_models/ensat.pt')
        accuracy_so_far = test_accuracy if test_accuracy > accuracy_so_far else accuracy_so_far
        print(f'Best accuracy so far: {accuracy_so_far:.4f}')


if __name__ == '__main__':
    main()
