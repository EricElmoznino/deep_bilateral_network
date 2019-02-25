from torch import nn


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
