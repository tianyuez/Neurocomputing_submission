"""test_util.py: utility for testing trained models."""
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm
import config
import torchattacks
from utils import GrayscaleToRgb


def test_util(
        model,
        model_gen_atk,
        device,
        epoch,
        writer,
        identifier,
        attack_type):

    if (config.dataset == 'FMNIST'):
        dataset = datasets.FashionMNIST(
            config.DATA_DIR / 'fmnist',
            train=False,
            download=True,
            transform=Compose(
                [
                    GrayscaleToRgb(),
                    ToTensor()]))
    if (config.dataset == 'SVHN'):
        dataset = datasets.SVHN(
            config.DATA_DIR / 'svhn', split='test', download=True, transform=Compose([ToTensor()]))
    elif (config.dataset == 'CIFAR10'):
        dataset = datasets.CIFAR10(
            config.DATA_DIR / 'cifar10', train=False, download=True, transform=Compose([ToTensor()]))
    elif (config.dataset == 'CIFAR100'):
        dataset = datasets.CIFAR100(
            config.DATA_DIR / 'cifar100', train=False, download=True, transform=Compose([ToTensor()]))

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True)
    total_accuracy = 0

    if (attack_type == 'FGSM'):
        attack = torchattacks.FGSM(model_gen_atk, eps=config.eps)
    elif (attack_type == 'PGD20'):
        attack = torchattacks.PGD(
            model_gen_atk,
            eps=config.eps,
            alpha=config.alpha,
            steps=20)
    elif (attack_type == 'PGD40'):
        attack = torchattacks.PGD(
            model_gen_atk,
            eps=config.eps,
            alpha=config.alpha,
            steps=40)
    elif (attack_type == 'MIFGSM'):
        attack = torchattacks.MIFGSM(model_gen_atk, eps=config.eps, steps=10)
    elif (attack_type == 'RFGSM'):
        attack = torchattacks.RFGSM(
            model_gen_atk,
            eps=config.eps,
            alpha=config.alpha)
    elif (attack_type == 'BIM'):
        attack = torchattacks.BIM(
            model_gen_atk,
            eps=config.eps,
            alpha=config.alpha)
    elif (attack_type == 'None'):
        attack = None
    for x, y_true in tqdm(dataloader, leave=False):
        if (attack_type == 'None'):
            x = x.to(device)
        else:
            x = attack(x, y_true)

        y_pred = model(x)
        total_accuracy += (y_pred.max(1)[1] ==
                           y_true.to(device)).float().mean().item()

    mean_accuracy = total_accuracy / len(dataloader)
    print(f'{identifier} Accuracy on target data: {mean_accuracy:.4f}')
    writer.add_scalar(f'{identifier} Accuracy/Epoch', mean_accuracy, epoch)
    writer.flush()

    return mean_accuracy
