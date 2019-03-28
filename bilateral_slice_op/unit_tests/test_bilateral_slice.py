import unittest
import torch
import bilateral_slice
import numpy as np


class BilateralSliceTests(unittest.TestCase):

    def test_bilateral_slice(self):
        grid = torch.from_numpy(np.load('grid.npy'))
        guide = torch.from_numpy(np.load('guide.npy'))
        input = torch.from_numpy(np.load('input.npy'))
        if torch.cuda.is_available():
            grid = grid.cuda()
            guide = guide.cuda()
            input = input.cuda()

        pytorch_output = bilateral_slice.forward(grid, guide, input, True).cpu().numpy()
        pytorch_output_no_offset = bilateral_slice.forward(grid, guide, input, False).cpu().numpy()

        tf_output = np.load('tf_output.npy').transpose([0, 3, 1, 2])
        tf_output_no_offset = np.load('tf_output_no_offset.npy').transpose([0, 3, 1, 2])

        print('pytorch_output shape:', pytorch_output.shape)
        print('pytorch_output_no_offset shape:', pytorch_output_no_offset.shape)
        print('tf_output shape:', tf_output.shape)
        print('tf_output_no_offset shape:', tf_output_no_offset.shape)

        difference = np.absolute(pytorch_output - tf_output)
        difference_no_offset = np.absolute(pytorch_output_no_offset - tf_output_no_offset)

        print('max abs:', np.absolute(tf_output).max())
        print('mean abs:', np.absolute(tf_output).mean())
        print('max difference:', difference.max())
        print('mean difference:', difference.mean())
        print('max difference_no_offset:', difference_no_offset.max())
        print('mean difference_no_offset:', difference_no_offset.mean())

        np.save('difference.npy', difference)
        np.save('difference_no_offset.npy', difference_no_offset)


if __name__ == '__main__':
    unittest.main()
