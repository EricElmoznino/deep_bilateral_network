from torch import nn
from models.layers import conv
from models.DeepBilateralNetCurves import DeepBilateralNetCurves


class DeepBilateralNetPointwiseNNGuide(DeepBilateralNetCurves):

    def forward_guidemap(self, image_fullres):
        conv1, conv2 = self.guide_params
        guidemap = conv1(image_fullres)
        guidemap = conv2(guidemap)
        guidemap = guidemap.squeeze(dim=1)
        return guidemap

    def make_guide_params(self):
        conv1 = conv(self.n_in, self.guide_pts, 1)
        conv2 = nn.Sequential(nn.Conv2d(self.guide_pts, 1, 1),
                              nn.BatchNorm2d(1),
                              nn.Sigmoid())
        return conv1, conv2
