import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class DeepBilateralNetCurves(nn.Module):

    def __init__(self, lowres_resolution, luma_bins, spatial_bin, channel_multiplier, n_in=3+1, n_out=3):
        super().__init__()
        self.luma_bins = luma_bins
        self.spatial_bin = spatial_bin
        self.channel_multiplier = channel_multiplier
        self.feature_multiplier = self.luma_bins * self.channel_multiplier
        self.n_in, self.n_out = n_in, n_out

        # coefficient model parameters
        self.splat = self.make_splat_features(lowres_resolution)
        self.global_conv, self.global_fc = self.make_global_features(self.splat[-1].shape[1])
        self.local = self.make_local_features(self.splat[-1].shape[1])
        self.prediction = conv(8 * self.feature_multiplier, luma_bins * n_in * n_out, 1, activation=False)

        # guide model parameters

        # output model parameters

    def forward(self, lowres_image, fullres_image):
        coefficients = self.forward_coefficients(lowres_image)

    def forward_coefficients(self, lowres_image):
        splat_features = self.splat(lowres_image)
        global_features = self.global_conv(splat_features)
        global_features = global_features.view(lowres_image.shape[0], global_features.shape[1] * 4 * 4)
        global_features = self.global_fc(global_features)
        global_features = global_features.view(lowres_image.shape[0], global_features.shape[1], 1, 1)
        local_features = self.local(splat_features)
        fusion = F.relu(global_features + local_features)
        coefficients = self.prediction(fusion)
        coefficients = torch.stack(torch.split(coefficients, self.luma_bins, dim=1), dim=2)
        coefficients = torch.stack(torch.split(coefficients, self.n_out, dim=2), dim=3)
        return coefficients

    def make_splat_features(self, lowres_resolution):
        splat_features = []
        in_channels = 3
        for i in range(int(np.log2(lowres_resolution / self.spatial_bin))):
            splat_features.append(conv(in_channels, (2 ** i) * self.feature_multiplier, 3, stride=2,
                                       batch_norm=False if i == 0 else True))
            in_channels = (2 ** i) * self.feature_multiplier
        splat_features = nn.Sequential(*splat_features)
        return splat_features

    def make_global_features(self, splat_channels):
        conv_features = []
        in_channels = splat_channels
        for _ in range(int(np.log2(self.spatial_bin / 4))):
            conv_features.append(conv(in_channels, 8 * self.feature_multiplier, 3, stride=2))
            in_channels = 8 * self.feature_multiplier
        conv_features = nn.Sequential(*conv_features)
        fc_features = nn.Sequential(fc(128 * self.feature_multiplier, 32 * self.feature_multiplier),
                                    fc(32 * self.feature_multiplier, 16 * self.feature_multiplier),
                                    fc(16 * self.feature_multiplier, 8 * self.feature_multiplier, activation=False))
        return conv_features, fc_features

    def make_local_features(self, splat_channels):
        local_features = nn.Sequential(conv(splat_channels, 8 * self.feature_multiplier, 3),
                                       conv(8 * self.feature_multiplier, 8 * self.feature_multiplier, 3,
                                            bias=False, activation=False))
        return local_features


def conv(in_channels, out_channels, kernel, stride=1, batch_norm=True, bias=True, activation=True):
    layers = [nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=(kernel - 1) // 2, bias=bias)]
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

