"""models_ensemble.py: pretrained models for ensemble adversarial training."""
import torch
from torch import nn


class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
            nn.Conv2d(3, 96, kernel_size=5, padding=2),
            nn.ELU(),

            nn.Conv2d(96, 96, kernel_size=1),
            nn.ELU(),

            nn.MaxPool2d(3, stride=2),

            nn.Dropout(),

            nn.Conv2d(96, 192, kernel_size=5, padding=2),
            nn.ELU(),

            nn.Conv2d(192, 192, kernel_size=1),
            nn.ELU(),

            nn.MaxPool2d(3, stride=2),

            nn.Dropout(0.5),

            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.ELU(),

            nn.Conv2d(256, 256, kernel_size=1),
            nn.ELU(),

            nn.Conv2d(256, 100, kernel_size=1),
            nn.AvgPool2d(7)
        )

    def forward(self, x):
        logits = self.net(x)
        logits = logits.squeeze()
        return logits


class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
            nn.Conv2d(3, 192, kernel_size=5, padding=2),
            nn.ELU(),

            nn.Conv2d(192, 96, kernel_size=1),
            nn.ELU(),

            nn.MaxPool2d(3, stride=2),

            nn.Dropout(),

            nn.Conv2d(96, 192, kernel_size=5, padding=2),
            nn.ELU(),

            nn.Conv2d(192, 192, kernel_size=1),
            nn.ELU(),

            nn.MaxPool2d(3, stride=2),

            nn.Dropout(0.5),

            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.ELU(),

            nn.Conv2d(256, 256, kernel_size=1),
            nn.ELU(),

            nn.Conv2d(256, 100, kernel_size=1),
            nn.AvgPool2d(7)
        )

    def forward(self, x):
        logits = self.net(x)
        logits = logits.squeeze()
        return logits


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std
