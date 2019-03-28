import torch

from models.layers import BilateralSliceFunction


def test_grad():
    grid_shape = torch.Size([1, 12, 8, 5, 3])
    guide_shape = torch.Size([1, 5, 3])
    img_shape = torch.Size([1, 3, 5, 3])

    grid = torch.randn(grid_shape)
    guide = torch.randn(guide_shape)
    img = torch.randn(img_shape)

    def bilateral_slice_offset(gr, gu, im):
        return BilateralSliceFunction.apply(gr, gu, im, True)

    variables = [grid, guide, img]
    for i, var in enumerate(variables):
        var = var.double()
        if torch.cuda.is_available():
            var = var.cuda()
        var.requires_grad = True
        variables[i] = var
    if torch.autograd.gradcheck(bilateral_slice_offset, variables, eps=1e-3, atol=1e-2):
        print('Ok')
    else:
        print('Not ok')


if __name__ == '__main__':
    test_grad()
