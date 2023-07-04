"""disc_f.py: feature-space discriminator."""
import torch.nn as nn
from utils import GradientReversal


class Disc_f(nn.Module):
    def __init__(self):
        super(Disc_f, self).__init__()

        self.disc = nn.Sequential(
            GradientReversal(),
            nn.Linear(12288, 500),
            nn.ReLU(),
            nn.Linear(500, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        out = self.disc(x)
        return out
