import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from models.util_layers import conv, fc


class DeepBilateralNetCurves(nn.Module):

    def __init__(self, lowres_resolution, fullres_resolution,
                 luma_bins, spatial_bin, channel_multiplier, n_in=3, n_out=3):
        super().__init__()
        self.luma_bins = luma_bins
        self.spatial_bin = spatial_bin
        self.channel_multiplier = channel_multiplier
        self.feature_multiplier = self.luma_bins * self.channel_multiplier
        self.n_in, self.n_out = n_in + 1, n_out

        # coefficient model parameters
        self.splat, self.global_conv, self.global_fc, self.local, self.prediction = \
            self.make_coefficient_params(lowres_resolution)

        # guide model parameters
        self.ccm, self.shifts, self.slopes, self.projection = self.make_guide_params()

        # output model parameters

    def forward(self, lowres_image, fullres_image):
        coefficients = self.forward_coefficients(lowres_image)
        guidemap = self.forward_guidemap(fullres_image)

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

    def forward_guidemap(self, fullres_image):
        guidemap = self.ccm(fullres_image)
        guidemap = guidemap.unsqueeze(dim=4)
        guidemap = (self.slopes * F.relu(guidemap - self.shifts)).sum(dim=4)
        guidemap = self.projection(guidemap)
        guidemap = guidemap.clamp(min=0, max=1)
        guidemap = guidemap.squeeze(dim=1)
        return guidemap

    def make_coefficient_params(self, lowres_resolution):
        # splat params
        splat = []
        in_channels = self.n_in - 1
        for i in range(int(np.log2(lowres_resolution / self.spatial_bin))):
            splat.append(conv(in_channels, (2 ** i) * self.feature_multiplier, 3, stride=2,
                              batch_norm=False if i == 0 else True))
            in_channels = (2 ** i) * self.feature_multiplier
        splat = nn.Sequential(*splat)
        splat_channels = in_channels

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
        n_ctr_pts = 16
        in_channels = self.n_in - 1

        ccm = conv(in_channels, in_channels, 1, batch_norm=False, activation=False,
                   weights_init=(np.identity(in_channels, dtype=np.float32) +
                                 np.random.randn(1).astype(np.float32) * 1e-4)
                   .reshape((in_channels, in_channels, 1, 1)),
                   bias_init=torch.zeros(in_channels))

        shifts = np.linspace(0, 1, n_ctr_pts, endpoint=False, dtype=np.float32)
        shifts = shifts[np.newaxis, np.newaxis, np.newaxis, :]
        shifts = np.tile(shifts, (in_channels, 1, 1, 1))
        shifts = nn.Parameter(data=torch.from_numpy(shifts))

        slopes = np.zeros([1, in_channels, 1, 1, n_ctr_pts], dtype=np.float32)
        slopes[:, :, :, :, 0] = 1.0
        slopes = nn.Parameter(data=torch.from_numpy(slopes))

        projection = conv(in_channels, 1, 1, activation=False, batch_norm=False,
                          weights_init=torch.ones(1, in_channels, 1, 1) / in_channels,
                          bias_init=torch.zeros(1))

        return ccm, shifts, slopes, projection
