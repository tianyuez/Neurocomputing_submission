"""models.py: the model to be trained and adapted."""
import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
            nn.Conv2d(3, 96, kernel_size=3, padding=1),
            nn.GroupNorm(32, 96),
            nn.ELU(),

            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.GroupNorm(32, 96),
            nn.ELU(),
            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 96),
            nn.ELU(),
            nn.Dropout(),

            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.GroupNorm(32, 192),
            nn.ELU(),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.GroupNorm(32, 192),
            nn.ELU(),
            nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 192),
            nn.ELU(),
            nn.Dropout(),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.GroupNorm(32, 192),
            nn.ELU(),
            nn.Conv2d(192, 192, kernel_size=1),
            nn.GroupNorm(32, 192),
            nn.ELU(),
            nn.Conv2d(192, 100, kernel_size=1),
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
