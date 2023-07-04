"""models_black.py: surrogate model for black-box attack."""
from torch import nn
import torch


class Net_b(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),
            nn.Conv2d(3, 96, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Dropout(0.2),

            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(96, 96, kernel_size=3, padding=1, stride=2),
            nn.ELU(),
            nn.Dropout(),

            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(192, 192, kernel_size=3, padding=1, stride=2),
            nn.ELU(),
            nn.Dropout(),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(192, 256, padding=1, kernel_size=3),
            nn.ELU(),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ELU(),
            nn.Conv2d(256, 10, kernel_size=1),
            nn.AvgPool2d(8)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
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
