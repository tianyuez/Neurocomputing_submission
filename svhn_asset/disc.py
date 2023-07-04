"""disc.py: logit-space discriminator."""
import torch.nn as nn
from utils import GradientReversal


class Disc(nn.Module):
    def __init__(self):
        super(Disc, self).__init__()

        self.disc = nn.Sequential(
            GradientReversal(),
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        out = self.disc(x)
        return out
