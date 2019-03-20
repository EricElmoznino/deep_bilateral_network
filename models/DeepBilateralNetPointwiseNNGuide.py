from torch import nn
from models.layers import conv
from models.DeepBilateralNetCurves import DeepBilateralNetCurves


class DeepBilateralNetPointwiseNNGuide(DeepBilateralNetCurves):

    def forward_guidemap(self, image_fullres):
        guidemap = self.guide_params.conv1(image_fullres)
        guidemap = self.guide_params.conv2(guidemap)
        guidemap = guidemap.squeeze(dim=1)
        return guidemap

    def make_guide_params(self):
        conv1 = conv(self.n_in, self.guide_pts, 1, norm=True)
        conv2 = nn.Sequential(nn.Conv2d(self.guide_pts, 1, 1),
                              nn.Sigmoid())
        guide_params = nn.Module()
        guide_params.conv1 = conv1
        guide_params.conv2 = conv2
        return guide_params
