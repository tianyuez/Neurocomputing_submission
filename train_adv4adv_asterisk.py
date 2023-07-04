"""train_adv4adv.py: Adv-4-Adv*."""
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
from test_util import test_util
from utils import GrayscaleToRgb

if(config.dataset == 'FMNIST'):
    from fmnist_asset.models import Net
    from fmnist_asset.models_black import Net_b
    from fmnist_asset.disc import Disc
    from fmnist_asset.disc_f import Disc_f
    model_path = Path('fmnist_asset/')
elif(config.dataset == 'SVHN'):
    from svhn_asset.models import Net
    from svhn_asset.models_black import Net_b
    from svhn_asset.disc import Disc
    from svhn_asset.disc_f import Disc_f
    model_path = Path('svhn_asset/')
elif(config.dataset == 'CIFAR10'):
    from cifar10_asset.models import Net
    from cifar10_asset.models_black import Net_b
    from cifar10_asset.disc import Disc
    from cifar10_asset.disc_f import Disc_f
    model_path = Path('cifar10_asset/')
elif(config.dataset == 'CIFAR100'):
    from cifar100_asset.models import Net
    from cifar100_asset.models_black import Net_b
    from cifar100_asset.disc import Disc
    from cifar100_asset.disc_f import Disc_f
    model_path = Path('cifar100_asset/')


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter()

    model = Net().to(device)
    model.load_state_dict(torch.load(model_path / 'trained_models/source.pt'))

    modelb = Net_b().to(device)
    modelb.load_state_dict(torch.load(model_path / 'trained_models/black.pt'))

    feature_extractor = model.feature_extractor
    clf = model.classifier

    # the list of logit-space discriminators
    discriminator_list = []
    discriminator_param = []

    # the list of feature-space discriminators
    discriminator_f_list = []
    discriminator_f_param = []

    if(config.dataset == 'FMNIST' or config.dataset == 'SVHN' or config.dataset == 'CIFAR10'):
        num_classes = 10
    elif(config.dataset == 'CIFAR100'):
        num_classes = 100

    # class-wise discriminators
    for i in range(num_classes):
        discriminator_list.append(Disc().to(device))
        discriminator_param = discriminator_param + \
            list(discriminator_list[i].parameters())

        discriminator_f_list.append(Disc_f().to(device))
        discriminator_f_param = discriminator_f_param + \
            list(discriminator_f_list[i].parameters())

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
        batch_size=128,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True)
    loader_target = DataLoader(
        dataset_training,
        batch_size=128,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True)

    optim = torch.optim.Adam(
        discriminator_param +
        discriminator_f_param +
        list(
            model.parameters()))

    accuracy_so_far = 0

    for epoch in range(1, config.num_epochs):
        batches = zip(loader_source, loader_target)
        n_batches = min(len(loader_source), len(loader_target))

        total_domain_loss = total_labels_accuracy = 0

        for (x_source, labels_source), (x_target, labels_target) in tqdm(
                batches, leave=False, total=n_batches):

            attack = torchattacks.FGSM(model, eps=config.eps)

            x_target = attack(x_target, labels_target)

            x = torch.cat([x_source.to(device), x_target])
            y_domain = torch.cat(
                [torch.ones(x_source.shape[0]), torch.zeros(x_target.shape[0])])
            labels = torch.cat([labels_source, labels_target])
            labels = labels.to(device)

            y_domain = y_domain.to(device)

            # get the representation in the feature space
            features = feature_extractor(x)
            if (config.dataset =='FMNIST' or config.dataset=='SVHN'):
                features = features.view(x.shape[0], -1)

            logits = clf(features).squeeze()

            if (config.dataset =='CIFAR10' or config.dataset=='CIFAR100'):
                features = features.view(x.shape[0], -1)            

            labels_loss = F.cross_entropy(
                logits[logits.shape[0] // 2:, :], labels[labels.shape[0] // 2:])

            domain_loss = []
            domain_loss_f = []

            # divide the samples by classes, and apply the class-wise
            # discriminators
            for i in range(num_classes):
                idx_k = (labels == (i + 1))
                logits_k = logits[idx_k, :]
                features_k = features[idx_k, :]
                y_domain_k = y_domain[idx_k]
                if (logits.shape[0] != 0):
                    domain_preds_k = discriminator_list[i](logits_k).squeeze()
                    domain_loss.append(
                        F.binary_cross_entropy_with_logits(
                            domain_preds_k.squeeze(), y_domain_k))

                    domain_preds_k_f = discriminator_f_list[i](
                        features_k).squeeze()
                    domain_loss_f.append(
                        F.binary_cross_entropy_with_logits(
                            domain_preds_k_f.squeeze(), y_domain_k))

                else:
                    domain_loss.append(0)
                    domain_loss_f.append(0)

            gamma = config.gamma
            beta = config.beta
            disc_loss = 0
            for i in range(num_classes):
                disc_loss = disc_loss + beta * \
                    domain_loss_f[i] + gamma * domain_loss[i]

            loss = labels_loss + disc_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_domain_loss += disc_loss.item()
            total_labels_accuracy += (logits.max(1)
                                      [1].squeeze() == labels).float().mean().item()

        # evaluate the performance of Adv-4-Adv using the PGD-20 attack
        test_accuracy = test_util(
            model,
            model,
            device,
            epoch,
            writer,
            'Adv-4-Adv',
            'PGD20')

        if test_accuracy > accuracy_so_far:
            torch.save(
                model.state_dict(),
                model_path / 'trained_models/adv4adva.pt')
        accuracy_so_far = test_accuracy if test_accuracy > accuracy_so_far else accuracy_so_far
        print(f'Best accuracy so far: {accuracy_so_far:.4f}')


if __name__ == '__main__':
    main()

