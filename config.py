"""config.py: configuration file."""
from pathlib import Path

dataset = 'CIFAR10'  # Please specify dataset
num_epochs = 400  # Please specify the number of epochs
DATA_DIR = Path('/') # Please specify the download location of the dataset
eps = 4/255# Please specify the max perturbation
alpha = 4/2550# Please specify the step size
gamma = 0.2# Please specify the coefficient for logit-space discriminators   
beta = 0# Please specify the coefficient for feature-space discriminators   
