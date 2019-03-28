import torch
from torch import nn
import numpy as np
import bilateral_slice


def conv(in_channels, out_channels, kernel, stride=1, norm=False, bias=True, relu=True,
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
    if norm:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def fc(in_features, out_features, norm=False, relu=True):
    layers = [nn.Linear(in_features, out_features)]
    if norm:
        layers.append(nn.BatchNorm1d(out_features))
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class BilateralSliceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bilateral_grid, guide, input, has_offset):
        ctx.save_for_backward(bilateral_grid, guide, input)
        ctx.has_offset = has_offset
        return bilateral_slice.forward(bilateral_grid, guide, input, has_offset)

    @staticmethod
    def backward(ctx, grad):
        bilateral_grid, guide, input = ctx.saved_variables
        d_grid, d_guide, d_input = bilateral_slice.backward(grad,
                                                            bilateral_grid,
                                                            guide,
                                                            input,
                                                            ctx.has_offset)
        return d_grid, d_guide, d_input, None


class BilateralSlice(torch.nn.Module):
    def __init__(self, has_offset):
        super().__init__()
        self.has_offset = has_offset

    def forward(self, bilateral_grid, guide, input):
        return BilateralSliceFunction(bilateral_grid, guide, input, self.has_offset)
