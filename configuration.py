import torch
from torch import optim
from torch.utils.data import DataLoader

from models.DeepBilateralNetCurves import DeepBilateralNetCurves
from datasets.BaseDataset import BaseDataset

# Class choice parameters
model_class = 'Curves'
dataset_class = 'Base'
data_dir = 'data/debug'
pretrained_path = None

# Model parameters
lowres = [256, 256]
highres = [512, 512]
luma_bins = 8
spatial_bins = 16
channel_multiplier = 1
guide_pts = 16

# Training parameters
n_epochs = 100
batch_size = 16
lr = 1e-4



