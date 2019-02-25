import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from models.util_layers import conv, fc


class DeepBilateralNetCurves(nn.Module):

    def __init__(self, lowres_resolution, fullres_resolution,
                 luma_bins, spatial_bin, channel_multiplier, n_in=3 + 1, n_out=3):
        super().__init__()
        self.luma_bins = luma_bins
        self.spatial_bin = spatial_bin
        self.channel_multiplier = channel_multiplier
        self.feature_multiplier = self.luma_bins * self.channel_multiplier
        self.n_in, self.n_out = n_in, n_out

        # coefficient model parameters
        self.splat, self.global_conv, self.global_fc, self.local, self.prediction = \
            self.make_coefficient_params(lowres_resolution)
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

    def make_coefficient_params(self, lowres_resolution):
        # splat params
        splat = []
        in_channels = self.n_in - 1
        for i in range(int(np.log2(lowres_resolution / self.spatial_bin))):
            splat.append(conv(in_channels, (2 ** i) * self.feature_multiplier, 3, stride=2,
                              batch_norm=False if i == 0 else True))
            in_channels = (2 ** i) * self.feature_multiplier
        splat = nn.Sequential(*splat)
        splat_channels = self.splat[-1].shape[1]

        # global params
        global_conv = []
        in_channels = splat_channels
        for _ in range(int(np.log2(self.spatial_bin / 4))):
            global_conv.append(conv(in_channels, 8 * self.feature_multiplier, 3, stride=2))
            in_channels = 8 * self.feature_multiplier
        global_conv = nn.Sequential(*global_conv)
        global_fc = nn.Sequential(fc(128 * self.feature_multiplier, 32 * self.feature_multiplier),
                                  fc(32 * self.feature_multiplier, 16 * self.feature_multiplier),
                                  fc(16 * self.feature_multiplier, 8 * self.feature_multiplier, activation=False))

        # local params
        local = nn.Sequential(conv(splat_channels, 8 * self.feature_multiplier, 3),
                              conv(8 * self.feature_multiplier, 8 * self.feature_multiplier, 3,
                                   bias=False, activation=False))

        # prediction params
        prediction = conv(8 * self.feature_multiplier, self.luma_bins * self.n_in * self.n_out, 1, activation=False)

        return splat, global_conv, global_fc, local, prediction

    def make_guide_params(self):
        pass
