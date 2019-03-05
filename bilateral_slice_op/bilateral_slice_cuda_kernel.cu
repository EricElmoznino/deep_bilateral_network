#include "bilateral_slice.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define DIV_UP(x, y) (((x) + (y) - 1)/(y))

__device__ float diff_abs(float x) {
  float eps = 1e-8;
  return sqrt(x*x+eps);
}

__device__ float d_diff_abs(float x) {
  float eps = 1e-8;
  return x/sqrt(x*x+eps);
}

__device__ float weight_z(float x) {
  float abx = diff_abs(x);
  return max(1.0f-abx, 0.0f);
}

__device__ float d_weight_z(float x) {
  float abx = diff_abs(x);
  if(abx > 1.0f) {
    return 0.0f;
    // return abx;
  } else {
    return d_diff_abs(x);
  }
}

template <typename scalar_t>
__global__ void
bilateral_slice_cuda_forward_kernel(scalar_t * __restrict__ output,
                                    scalar_t * __restrict__ bilateral_grid,
                                    scalar_t * __restrict__ guide,
                                    scalar_t * __restrict__ input,
                                    GridSizes gsz,
                                    bool has_offset,
                                    int total_count,
                                    int output_chans)
{
        int h = gsz.h;
        int w = gsz.w;
        int gd = gsz.gd;
        int gh = gsz.gh;
        int gw = gsz.gw;
        int input_chans = gsz.input_chans;
        int coeff_stride = input_chans;
        int grid_chans = input_chans*output_chans;

        for (int idx = blockIdx.x*blockDim.x + threadIdx.x;
             idx < total_count;
             idx += blockDim.x*gridDim.x) {
                int x = idx % w;
                int y = (idx / w) % h;
                int out_c = (idx / (h*w)) % output_chans;
                int b = (idx / (output_chans*w*h));

                float gx = (x + 0.5f)*gw/(1.0f*w);
                float gy = (y + 0.5f)*gh/(1.0f*h);
                float gz = guide[x + w*(y + h*b)]*gd;

                int fx = static_cast<int>(floor(gx - 0.5f));
                int fy = static_cast<int>(floor(gy - 0.5f));
                int fz = static_cast<int>(floor(gz - 0.5f));


                // Grid strides
                int sz = grid_chans;
                int sx = grid_chans*gd;
                int sy = grid_chans*gd*gw;
                int sb = grid_chans*gd*gw*gh;

                float value = 0.0f;
                for (int in_c = 0;
                     in_c < coeff_stride;
                     ++in_c) {
                        float coeff_sample = 0.0f;

                        for (int xx = fx; xx < fx + 2; ++xx) {
                                int x_ = max(min(xx, gw - 1), 0);
                                float wx = max(1.0f - abs(xx + 0.5 - gx), 0.0f);

                                for (int yy = fy; yy < fy + 2; ++yy) {
                                        int y_ = max(min(yy, gh - 1), 0);
                                        float wy = max(1.0f - abs(yy + 0.5 - gy), 0.0f);

                                        for (int zz = fz; zz < fz + 2; ++zz) {
                                                int z_ = max(min(zz, gd - 1), 0);
                                                float wz = weight_z(zz + 0.5 - gz);
                                                int grid_idx = ((coeff_stride*out_c + in_c) +
                                                                sz*z_ + sx*x_ + sy*y_ + sb*b);

                                                coeff_sample += bilateral_grid[grid_idx]*wx*wy*wz;
                                        }
                                }
                        } // Grid trilinear interpolation
                        if (in_c < input_chans) {
                                int input_idx = in_c + input_chans*(x + w*(y + h*b));
                                value += coeff_sample*input[input_idx];
                        } else { // Offset term
                                value += coeff_sample;
                        }
                }

                output[idx] = value;
        }
}

at::Tensor
bilateral_slice_cuda_forward(at::Tensor output_tensor,
                             at::Tensor bilateral_grid,
                             at::Tensor guide,
                             at::Tensor input,
                             GridSizes& gsz,
                             bool has_offset)
{
        int total_count = gsz.bs*gsz.h*gsz.w*output_tensor.sizes()[1];
        int threads = 1024;
        cudaDeviceProp *prop = at::cuda::getCurrentDeviceProperties();
        int physical_thread_count =
                std::min((prop->multiProcessorCount *
                          prop->maxThreadsPerMultiProcessor),
                         total_count);
        int blocks = std::min(DIV_UP(physical_thread_count, threads),
                              prop->multiProcessorCount);

        AT_DISPATCH_FLOATING_TYPES(
                guide.type(),
                "bilateral_slice_cuda_forward",
                ([&] {
                 bilateral_slice_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
                         output_tensor.data<scalar_t>(),
                         bilateral_grid.data<scalar_t>(),
                         guide.data<scalar_t>(),
                         input.data<scalar_t>(),
                         gsz,
                         has_offset,
                         total_count,
                         output_tensor.sizes()[1]);
                 }));

        return output_tensor;
}

at::Tensor
bilateral_slice_cuda_backward(at::Tensor grad,
                              at::Tensor bilateral_grid,
                              at::Tensor guide,
                              at::Tensor input,
                              bool has_offset)
{
        return input;
}
