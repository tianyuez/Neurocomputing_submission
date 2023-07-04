"""train_atda.py: ATDA."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor

import config
from test_util import test_util
from utils import GrayscaleToRgb
from pathlib import Path

from tqdm import tqdm
import torchattacks

from loss import coral
from loss import mmd
from loss import MarginLoss

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


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter()

    model = Net().to(device)
    model.load_state_dict(torch.load(model_path / 'trained_models/source.pt'))

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
    loader_target = DataLoader(
        dataset_training,
        batch_size=256,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True)

    optim = torch.optim.Adam(list(model.parameters()))

    accuracy_so_far = 0

    if(config.dataset == 'FMNIST' or config.dataset == 'SVHN' or config.dataset == 'CIFAR10'):
        margin = MarginLoss()
    elif(config.dataset == 'CIFAR100'):
        margin = MarginLoss(num_classes=100, feat_dim=100)

    for epoch in range(1, config.num_epochs):
        batches = zip(loader_source, loader_target)
        n_batches = min(len(loader_source), len(loader_target))

        for (x_source, labels_source), (x_target, labels_target) in tqdm(
                batches, leave=False, total=n_batches):

            attack = torchattacks.FGSM(model, eps=config.eps)

            x_target = attack(x_target, labels_target)

            x = torch.cat([x_source.to(device), x_target])

            labels = torch.cat([labels_source, labels_target])
            labels = labels.to(device)

            logits = model(x).squeeze()

            logits_source = model(x_source.to(device))
            logits_target = model(x_target)

            labels_loss_source = F.cross_entropy(
                logits_source, labels_source.to(device))
            labels_loss_target = F.cross_entropy(
                logits_target, labels_target.to(device))

            coral_loss = coral(logits_source, logits_target)
            mmd_loss = mmd(logits_source, logits_target)
            margin_loss = margin(logits, labels)

            loss = labels_loss_source + labels_loss_target + \
                (coral_loss + mmd_loss + margin_loss) / 3

            optim.zero_grad()
            loss.backward()
            optim.step()

        test_accuracy = test_util(
            model, model, device, epoch, writer, 'ATDA', 'PGD20')

        if test_accuracy > accuracy_so_far:
            torch.save(
                model.state_dict(),
                model_path / 'trained_models/atda.pt')
        accuracy_so_far = test_accuracy if test_accuracy > accuracy_so_far else accuracy_so_far
        print(f'Best accuracy so far: {accuracy_so_far:.4f}')


if __name__ == '__main__':
    main()
