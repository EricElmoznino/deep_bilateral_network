import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from models.layers import conv, fc, BilateralSliceFunction


class DeepBilateralNetCurves(nn.Module):

    def __init__(self, lowres, luma_bins, spatial_bin, channel_multiplier, guide_pts, norm=False, n_in=3, n_out=3,
                 iteratively_upsample=False):
        super().__init__()
        self.luma_bins = luma_bins
        self.spatial_bin = spatial_bin
        self.channel_multiplier = channel_multiplier
        self.feature_multiplier = self.luma_bins * self.channel_multiplier
        self.guide_pts = guide_pts
        self.norm = norm
        self.n_in, self.n_out = n_in, n_out
        self.iteratively_upsample = iteratively_upsample

        self.coefficient_params = self.make_coefficient_params(lowres)
        self.guide_params = self.make_guide_params()

    def forward(self, image_lowres, image_fullres):
        coefficients = self.forward_coefficients(image_lowres)
        if self.iteratively_upsample:
            coefficients = self.iterative_upsample(coefficients, image_fullres.shape[2:4])
        guidemap = self.forward_guidemap(image_fullres)
        output = BilateralSliceFunction.apply(coefficients, guidemap, image_fullres, True)
        if not self.training:
            output = F.hardtanh(output, min_val=0, max_val=1)
        return output

    def forward_coefficients(self, image_lowres):
        splat_features = self.coefficient_params.splat(image_lowres)
        global_features = self.coefficient_params.global_conv(splat_features)
        global_features = global_features.view(image_lowres.shape[0], -1)
        global_features = self.coefficient_params.global_fc(global_features)
        global_features = global_features.view(image_lowres.shape[0], global_features.shape[1], 1, 1)
        local_features = self.coefficient_params.local(splat_features)
        fusion = F.relu(global_features + local_features)
        coefficients = self.coefficient_params.prediction(fusion)
        coefficients = torch.stack(torch.split(coefficients, self.n_out*(self.n_in+1), dim=1), dim=2)
        return coefficients

    def forward_guidemap(self, image_fullres):
        guidemap = self.guide_params.ccm(image_fullres)
        guidemap = guidemap.unsqueeze(dim=4)
        guidemap = (self.guide_params.slopes * F.relu(guidemap - self.guide_params.shifts)).sum(dim=4)
        guidemap = self.guide_params.projection(guidemap)
        guidemap = F.hardtanh(guidemap, min_val=0, max_val=1)
        guidemap = guidemap.squeeze(dim=1)
        return guidemap

    def make_coefficient_params(self, lowres):
        # splat params
        splat = []
        in_channels = self.n_in
        num_downsamples = int(np.log2(min(lowres) / self.spatial_bin))
        extra_convs = max(0, int(np.log2(self.spatial_bin) - np.log2(16)))
        extra_convs = np.linspace(0, num_downsamples - 1, extra_convs, dtype=np.int).tolist()
        for i in range(num_downsamples):
            out_channels = (2 ** i) * self.feature_multiplier
            splat.append(conv(in_channels, out_channels, 3, stride=2, norm=False if i == 0 else self.norm))
            if i in extra_convs:
                splat.append(conv(out_channels, out_channels, 3, norm=self.norm))
            in_channels = out_channels
        splat = nn.Sequential(*splat)
        splat_channels = in_channels

        # global params
        global_conv = []
        in_channels = splat_channels
        for _ in range(int(np.log2(self.spatial_bin / 4))):
            global_conv.append(conv(in_channels, 8 * self.feature_multiplier, 3, stride=2, norm=self.norm))
            in_channels = 8 * self.feature_multiplier
        global_conv.append(nn.AdaptiveAvgPool2d(4))
        global_conv = nn.Sequential(*global_conv)
        global_fc = nn.Sequential(fc(128 * self.feature_multiplier, 32 * self.feature_multiplier, norm=self.norm),
                                  fc(32 * self.feature_multiplier, 16 * self.feature_multiplier, norm=self.norm),
                                  fc(16 * self.feature_multiplier, 8 * self.feature_multiplier, norm=False, relu=False))

        # local params
        local = nn.Sequential(conv(splat_channels, 8 * self.feature_multiplier, 3),
                              conv(8 * self.feature_multiplier, 8 * self.feature_multiplier, 3,
                                   bias=False, norm=False, relu=False))

        # prediction params
        prediction = conv(8 * self.feature_multiplier, self.luma_bins * (self.n_in+1) * self.n_out, 1,
                          norm=False, relu=False)

        coefficient_params = nn.Module()
        coefficient_params.splat = splat
        coefficient_params.global_conv = global_conv
        coefficient_params.global_fc = global_fc
        coefficient_params.local = local
        coefficient_params.prediction = prediction
        return coefficient_params

    def make_guide_params(self):
        ccm = conv(self.n_in, self.n_in, 1, norm=False, relu=False,
                   weights_init=(np.identity(self.n_in, dtype=np.float32) +
                                 np.random.randn(1).astype(np.float32) * 1e-4)
                   .reshape((self.n_in, self.n_in, 1, 1)),
                   bias_init=torch.zeros(self.n_in))

        shifts = np.linspace(0, 1, self.guide_pts, endpoint=False, dtype=np.float32)
        shifts = shifts[np.newaxis, np.newaxis, np.newaxis, :]
        shifts = np.tile(shifts, (self.n_in, 1, 1, 1))
        shifts = nn.Parameter(data=torch.from_numpy(shifts))

        slopes = np.zeros([1, self.n_in, 1, 1, self.guide_pts], dtype=np.float32)
        slopes[:, :, :, :, 0] = 1.0
        slopes = nn.Parameter(data=torch.from_numpy(slopes))

        projection = conv(self.n_in, 1, 1, norm=False, relu=False,
                          weights_init=torch.ones(1, self.n_in, 1, 1) / self.n_in,
                          bias_init=torch.zeros(1))

        guide_params = nn.Module()
        guide_params.ccm = ccm
        guide_params.shifts = shifts
        guide_params.slopes = slopes
        guide_params.projection = projection
        return guide_params

    def iterative_upsample(self, coefficients, fullres):
        res = self.spatial_bin
        coefficients = coefficients.view(coefficients.shape[0], -1, res, res)
        while res * 2 < min(fullres):
            res *= 2
            coefficients = F.upsample(coefficients, size=[res, res], mode='bilinear')
        coefficients = coefficients.view(coefficients.shape[0], -1, self.luma_bins, res, res)
        return coefficients
