import torch
import bilateral_slice_cuda


def test_bilateral_slice():
    batch_size = 3
    grid_shape = [batch_size, 12, 8, 10, 6]
    guide_shape = [batch_size, 101, 60]
    input_shape = [batch_size, 3, 101, 60]

    grid = torch.zeros(grid_shape).cuda()
    guide = torch.zeros(guide_shape).cuda()
    img = torch.zeros(input_shape).cuda()

    x = bilateral_slice_cuda.forward(grid, guide, img, True)
    print(x)
    x = bilateral_slice_cuda.forward(grid, guide, img, False)
    print(x)


if __name__ == '__main__':
    test_bilateral_slice()
