"""test.py: test file."""
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import config
from test_util import test_util

from pathlib import Path


if(config.dataset == 'FMNIST'):
    from fmnist_asset.models import Net
    from fmnist_asset.models_black import Net_b
    model_path = Path('fmnist_asset/')
elif(config.dataset == 'SVHN'):
    from svhn_asset.models import Net
    from svhn_asset.models_black import Net_b
    model_path = Path('svhn_asset/')
elif(config.dataset == 'CIFAR10'):
    from cifar10_asset.models import Net
    from cifar10_asset.models_black import Net_b
    model_path = Path('cifar10_asset/')
elif(config.dataset == 'CIFAR100'):
    from cifar100_asset.models import Net
    from cifar100_asset.models_black import Net_b
    model_path = Path('cifar100_asset/')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter()

    model_adv4adv = Net().to(device)
    model_adv4adv.load_state_dict(
        torch.load(
            model_path /
            'trained_models/adv4adv.pt'))

    model_black = Net_b().to(device)
    model_black.load_state_dict(
        torch.load(
            model_path /
            'trained_models/black.pt'))

    epoch = None

    # Example: test the performance of model trained by Adv-4-Adv under
    # white-box PGD-20 attack
    test_util(
        model_adv4adv,
        model_adv4adv,
        device,
        epoch,
        writer,
        'W-Adv4Adv-PGD20',
        'PGD20')


if __name__ == '__main__':
    main()
