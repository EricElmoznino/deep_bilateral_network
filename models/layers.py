import torch
from torch import nn
import numpy as np


def conv(in_channels, out_channels, kernel, stride=1, batch_norm=True, bias=True, activation=True,
         weights_init=None, bias_init=None):
    layers = [nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=(kernel - 1) // 2, bias=bias)]
    assert not (bias_init is not None and not bias)
    if weights_init is not None:
        if isinstance(weights_init, np.ndarray):
            weights_init = torch.from_numpy(weights_init)
        layers[0].weight.data = weights_init
    if bias_init is not None:
        if isinstance(bias_init, np.ndarray):
            bias_init = torch.from_numpy(bias_init)
        layers[0].bias.data = bias_init
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    if activation:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def fc(in_features, out_features, batch_norm=True, activation=True):
    layers = [nn.Linear(in_features, out_features)]
    if batch_norm:
        layers.append(nn.BatchNorm1d(out_features))
    if activation:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)
