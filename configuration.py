import torch
from torch import optim
from torch.utils.data import DataLoader
import os
from models.DeepBilateralNetCurves import DeepBilateralNetCurves
from models.DeepBilateralNetPointwiseNNGuide import DeepBilateralNetPointwiseNNGuide
from datasets.BaseDataset import BaseDataset
import utils

# Class choice parameters
model_class = 'Curves'
dataset_class = 'Base'
data_dir = 'data/hair_blended'
pretrained_path = None

# Model parameters
lowres = [256, 256]
fullres = [512, 512]
luma_bins = 4
spatial_bins = 64
channel_multiplier = 1
guide_pts = 4

# Training parameters
n_epochs = 100
batch_size = 4
lr = 1e-4


def params():
    train_loader, test_loader = get_dataloaders()
    model = get_model()
    optimizer, scheduler = get_optimizer(model)
    return n_epochs, train_loader, test_loader, model, optimizer, scheduler


def get_dataloaders():
    if dataset_class == 'Base':
        train_set = BaseDataset(os.path.join(data_dir, 'train'), lowres, fullres, training=True)
        test_set = BaseDataset(os.path.join(data_dir, 'test'), lowres, fullres)
    else:
        raise NotImplementedError()
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=2, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=2)
    return train_loader, test_loader


def get_model():
    if model_class == 'Curves':
        model = DeepBilateralNetCurves(lowres, luma_bins, spatial_bins, channel_multiplier, guide_pts)
    elif model_class == 'NN':
        model = DeepBilateralNetPointwiseNNGuide(lowres, luma_bins, spatial_bins, channel_multiplier, guide_pts)
    else:
        raise NotImplementedError()
    if torch.cuda.is_available():
        model.cuda()
    if pretrained_path is not None:
        utils.load_model(model, pretrained_path)
    return model


def get_optimizer(model):
    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[n_epochs // 2 + i * 10 for i in range(10)],
                                               gamma=0.5)
    return optimizer, scheduler
