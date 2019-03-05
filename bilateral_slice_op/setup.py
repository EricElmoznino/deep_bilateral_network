from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(name='bilateral_slice_op',
      ext_modules=[
          CUDAExtension('bilateral_slice_cuda', [
              'bilateral_slice_cuda.cpp',
              'bilateral_slice_cuda_kernel.cu',
          ])
      ],
      cmdclass={
          'build_ext': BuildExtension
      })
