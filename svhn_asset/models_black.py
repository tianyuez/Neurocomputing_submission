"""models_black.py: surrogate model for black-box attack."""
from torch import nn
import torch


class Net_b(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            Normalize(
                mean=[
                    0.4376821, 0.4437697, 0.47280442], std=[
                    0.19803012, 0.20101562, 0.19703614]), nn.Conv2d(
                3, 64, kernel_size=3, stride=1), nn.ReLU(), )

        self.classifier = nn.Sequential(
            nn.Linear(57600, 300),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(300, 10)
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
