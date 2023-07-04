"""models_ensemble.py: pretrained models for ensemble adversarial training."""
import torch
from torch import nn


class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            Normalize(
                mean=[
                    0.4376821, 0.4437697, 0.47280442], std=[
                    0.19803012, 0.20101562, 0.19703614]), nn.Conv2d(
                3, 32, kernel_size=5, stride=2), nn.ReLU(), nn.Dropout(0.1), )

        self.classifier = nn.Sequential(
            nn.Linear(6272, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(x.shape[0], -1)
        logits = self.classifier(features)
        logits = logits.squeeze()
        return logits


class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            Normalize(
                mean=[
                    0.4376821, 0.4437697, 0.47280442], std=[
                    0.19803012, 0.20101562, 0.19703614]), nn.Dropout(0.2), nn.Conv2d(
                3, 32, kernel_size=3, stride=2), nn.ReLU(), nn.Conv2d(
                        32, 32, kernel_size=3, stride=2), nn.ReLU(), )

        self.classifier = nn.Sequential(
            nn.Linear(1568, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(x.shape[0], -1)
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
