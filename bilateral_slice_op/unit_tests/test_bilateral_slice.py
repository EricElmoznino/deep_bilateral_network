import unittest
import torch
import bilateral_slice_cuda
import numpy as np


class BilateralSliceTests(unittest.TestCase):

    def test_bilateral_slice(self):
        batch_size = 3
        grid_shape = [batch_size, 12, 8, 10, 6]
        guide_shape = [batch_size, 101, 60]
        input_shape = [batch_size, 3, 101, 60]

        grid = torch.ones(grid_shape).cuda() * 0.1
        guide = torch.ones(guide_shape).cuda() * 0.1
        img = torch.ones(input_shape).cuda() * 0.1

        pytorch_output = bilateral_slice_cuda.forward(grid, guide, img, True).cpu().numpy()
        pytorch_output_no_offset = bilateral_slice_cuda.forward(grid, guide, img, False).cpu().numpy()

        tf_output = np.load('tf_output.npy').transpose([0, 3, 1, 2])
        tf_output_no_offset = np.load('tf_output_no_offset.npy').transpose([0, 3, 1, 2])

        print('pytorch_output shape:', pytorch_output.shape)
        print('pytorch_output_no_offset shape:', pytorch_output_no_offset.shape)
        print('tf_output shape:', tf_output.shape)
        print('tf_output_no_offset shape:', tf_output_no_offset.shape)

        difference = np.absolute(pytorch_output - tf_output)
        difference_no_offset = np.absolute(pytorch_output_no_offset - tf_output_no_offset)

        print('max difference:', difference.max())
        print('mean difference:', difference.mean())
        print('max difference_no_offset:', difference_no_offset.max())
        print('mean difference_no_offset:', difference_no_offset.mean())

        np.save('difference.npy', difference)
        np.save('difference_no_offset.npy', difference_no_offset)


if __name__ == '__main__':
    unittest.main()
