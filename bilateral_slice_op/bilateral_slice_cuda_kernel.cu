#include "bilateral_slice.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define DIV_UP(x, y) (((x) + (y) - 1)/(y))

struct DevProp {
        int blocks;
        int threads;
};

static DevProp
get_thread_blocks(int total_count)
{
        int threads = 1024;
        cudaDeviceProp *prop = at::cuda::getCurrentDeviceProperties();
        int physical_thread_count =
                std::min((prop->multiProcessorCount *
                          prop->maxThreadsPerMultiProcessor),
                         total_count);
        int blocks = std::min(DIV_UP(physical_thread_count, threads),
                              prop->multiProcessorCount);

        return {blocks, threads};
}

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
  return max(1.0f - abx, 0.0f);
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

        if (has_offset) {
                grid_chans += output_chans;
                coeff_stride += 1;
        }

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
                int sy = gw;
                int sz = gw*gh;
                int sc = gd*gw*gh;
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
                                                int c_ = coeff_stride*out_c + in_c;
                                                int grid_idx = x_ + sy*y_ + sz*z_ + sc*c_ + sb*b;

                                                coeff_sample += bilateral_grid[grid_idx]*wx*wy*wz;
                                        }
                                }
                        } // Grid trilinear interpolation
                        if (in_c < input_chans) {
                                int input_idx = x + w*(y + h*(in_c + input_chans*b));
                                value += coeff_sample*input[input_idx];
                        } else { // Offset term
                                value += coeff_sample;
                        }
                }

                output[idx] = value;
        }
}

template <typename scalar_t>
__global__ void
bilateral_slice_cuda_grid_grad_kernel(scalar_t * __restrict__ out_grid_grad,
                                      scalar_t * __restrict__ upstream_grad,
                                      scalar_t * __restrict__ guide,
                                      scalar_t * __restrict__ input,
                                      GridSizes gsz,
                                      bool has_offset,
                                      int grid_count,
                                      int output_chans)
{
        int h = gsz.h;
        int w = gsz.w;
        int gd = gsz.gd;
        int gh = gsz.gh;
        int gw = gsz.gw;
        int input_chans = gsz.input_chans;
        int grid_chans = input_chans*output_chans;
        int coeff_stride = input_chans;

        if (has_offset) {
                grid_chans += output_chans;
                coeff_stride += 1;
        }

        for (int idx = blockIdx.x*blockDim.x + threadIdx.x;
             idx < grid_count;
             idx += blockDim.x*gridDim.x) {
                int gx = idx % gw;
                int gy = (idx/gw) % gh;
                int gz = (idx/(gh*gw)) % gd;
                int c = (idx/(gd*gh*gw)) % grid_chans;
                int b = (idx/(grid_chans*gd*gw*gh));

                float scale_w = w*1.0/gw;
                float scale_h = h*1.0/gh;

                int left_x = static_cast<int>(floor(scale_w*(gx + 0.5 - 1)));
                int right_x = static_cast<int>(ceil(scale_w*(gx + 0.5 + 1)));
                int left_y = static_cast<int>(floor(scale_h*(gy + 0.5 - 1)));
                int right_y = static_cast<int>(ceil(scale_h*(gy + 0.5 + 1)));

                // Strides in the output
                int sy = w;
                int sc = w*h;
                int sb = output_chans*w*h;

                // Strides in the input
                int isy = w;
                int isc = h*w;
                int isb = input_chans*h*w;

                int out_c = c / coeff_stride;
                int in_c = c % coeff_stride;

                float value = 0.0f;
                for (int x = left_x;
                     x < right_x;
                     ++x) {
                        int x_ = x;

                        // mirror boundary
                        if (x_ < 0)
                                x_ = -x_ - 1;
                        if (x_ >= w)
                                x_ = 2*w - 1 - x_;

                        float gx2 = (x + 0.5f)/scale_w;
                        float wx = max(1.0f - abs(gx + 0.5 - gx2), 0.0f);

                        for (int y = left_y;
                             y < right_y;
                             ++y) {
                                int y_ = y;

                                // mirror boundary
                                if (y_ < 0)
                                        y_ = -y_ - 1;
                                if (y_ >= h)
                                        y_ = 2*h - 1 - y_;

                                float gy2 = (y + 0.5f)/scale_h;
                                float wy = max(1.0f - abs(gy + 0.5 - gy2), 0.0f);

                                int guide_idx = x_ + w*y_ + h*w*b;
                                float gz2 = guide[guide_idx]*gd;
                                float wz = weight_z(gz + 0.5f - gz2);
                                if (((gz == 0) && (gz2 < 0.5f)) ||
                                    ((gz == (gd - 1)) && (gz2 > (gd - 0.5f))))
                                        wz = 1.0f;

                                int back_idx = x_ + sy*y_ + sc*out_c + sb*b;
                                if (in_c < input_chans) {
                                        int input_idx = x_ + isy*y_ + isc*in_c + isb*b;
                                        value += wz*wx*wy*upstream_grad[back_idx]*input[input_idx];
                                } else { // offset term
                                        value += wz*wx*wy*upstream_grad[back_idx];
                                }
                        }
                }

                out_grid_grad[idx] = value;
        }
}

template <typename scalar_t>
__global__ void
bilateral_slice_cuda_guide_grad_kernel(scalar_t * __restrict__ out_guide_grad,
                                       scalar_t * __restrict__ upstream_grad,
                                       scalar_t * __restrict__ bilateral_grid,
                                       scalar_t * __restrict__ guide,
                                       scalar_t * __restrict__ input,
                                       GridSizes gsz,
                                       bool has_offset,
                                       int guide_count,
                                       int output_chans)
{
        int h = gsz.h;
        int w = gsz.w;
        int gd = gsz.gd;
        int gh = gsz.gh;
        int gw = gsz.gw;
        int input_chans = gsz.input_chans;
        int grid_chans = input_chans*output_chans;
        int coeff_stride = input_chans;

        if (has_offset) {
                grid_chans += output_chans;
                coeff_stride += 1;
        }

        for (int idx = blockIdx.x*blockDim.x + threadIdx.x;
             idx < guide_count;
             idx += blockDim.x*gridDim.x) {
                int x = idx  % w;
                int y = (idx / w) % h;
                int b = (idx / (w*h));

                float gx = (x + 0.5f)*gw/(1.0f*w);
                float gy = (y + 0.5f)*gh/(1.0f*h);
                float gz = guide[x + w*(y + h*b)]*gd;

                int fx = static_cast<int>(floor(gx - 0.5f));
                int fy = static_cast<int>(floor(gy - 0.5f));
                int fz = static_cast<int>(floor(gz - 0.5f));

                // Grid stride
                int sy = gw;
                int sz = gh*gw;
                int sc = gd*gh*gw;
                int sb = grid_chans*gd*gw*gh;

                float out_sum = 0.0f;
                for (int out_c = 0;
                     out_c < output_chans;
                     ++out_c) {

                        float in_sum = 0.0f;
                        for (int in_c = 0;
                             in_c < coeff_stride;
                             ++in_c) {

                                float grid_sum = 0.0f;
                                for (int xx = fx;
                                     xx < fx + 2;
                                     ++xx) {
                                        int x_ = max(min(xx, gw - 1), 0);
                                        float wx = max(1.0f - abs(xx + 0.5 - gx), 0.0f);

                                        for (int yy = fy;
                                             yy < fy + 2;
                                             ++yy) {
                                                int y_ = max(min(yy, gh - 1), 0);
                                                float wy = max(1.0f - abs(yy + 0.5 - gy), 0.0f);

                                                for (int zz = fz;
                                                     zz < fz + 2;
                                                     ++zz) {
                                                        int z_ = max(min(zz, gd - 1), 0);
                                                        float dwz = gd*d_weight_z(zz + 0.5 - gz);

                                                        int c_ = coeff_stride*out_c + in_c;
                                                        int grid_idx = x_ + sy*y_ + sz*z_ + sc*c_ + sb*b;
                                                        grid_sum += bilateral_grid[grid_idx]*wx*wy*dwz;
                                                } // z
                                        } // y
                                } // x, grid trilinear interp

                                if (in_c < input_chans)
                                        in_sum += grid_sum*input[x + w*(y + h*(in_c + input_chans*b))];
                                else  // offset term
                                        in_sum += grid_sum;
                        } // in_c

                        out_sum += in_sum*upstream_grad[x + w*(y + h*(out_c + output_chans*b))];
                } // out_c

                out_guide_grad[idx] = out_sum;
        }
}

template <typename scalar_t>
__global__ void
bilateral_slice_cuda_input_grad_kernel(scalar_t * __restrict__ out_input_grad,
                                       scalar_t * __restrict__ upstream_grad,
                                       scalar_t * __restrict__ bilateral_grid,
                                       scalar_t * __restrict__ guide,
                                       GridSizes gsz,
                                       bool has_offset,
                                       int input_count,
                                       int output_chans)
{
        int h = gsz.h;
        int w = gsz.w;
        int gd = gsz.gd;
        int gh = gsz.gh;
        int gw = gsz.gw;
        int input_chans = gsz.input_chans;
        int grid_chans = input_chans*output_chans;
        int coeff_stride = input_chans;

        if (has_offset) {
                grid_chans += output_chans;
                coeff_stride += 1;
        }

        for (int idx = blockIdx.x*blockDim.x + threadIdx.x;
             idx < input_count;
             idx += blockDim.x*gridDim.x) {
                int x = idx % w;
                int y = (idx / w) % h;
                int in_c = (idx / (h*w)) % input_chans;
                int b = (idx / (input_chans*w*h));

                float gx = (x + 0.5f)*gw/(1.0f*w);
                float gy = (y + 0.5f)*gh/(1.0f*h);
                float gz = guide[x + w*(y + h*b)]*gd;

                int fx = static_cast<int>(floor(gx - 0.5f));
                int fy = static_cast<int>(floor(gy - 0.5f));
                int fz = static_cast<int>(floor(gz - 0.5f));

                // Grid stride
                int sy = gw;
                int sz = gh*gw;
                int sc = gd*gh*gw;
                int sb = grid_chans*gd*gh*gw;

                float value = 0.0f;
                for (int out_c = 0;
                     out_c < output_chans;
                     ++out_c) {
                        float chan_val = 0.0f;

                        for (int xx = fx;
                             xx < fx+2;
                             ++xx) {
                                int x_ = max(min(xx, gw - 1), 0);
                                float wx = max(1.0f - abs(xx + 0.5 - gx), 0.0f);

                                for (int yy = fy;
                                     yy < fy + 2;
                                     ++yy) {
                                        int y_ = max(min(yy, gh - 1), 0);
                                        float wy = max(1.0f - abs(yy + 0.5 - gy), 0.0f);

                                        for (int zz = fz;
                                             zz < fz+2;
                                             ++zz) {

                                                int z_ = max(min(zz, gd - 1), 0);

                                                float wz = weight_z(zz + 0.5 - gz);

                                                int c_ = coeff_stride*out_c + in_c;
                                                int grid_idx = x_ + sy*y_ + sz*z_ + sc*c_ + sb*b;
                                                chan_val += bilateral_grid[grid_idx]*wx*wy*wz;
                                        } // z
                                } // y
                        } // x, grid trilinear interp

                        value += chan_val*upstream_grad[x + w*(y + h*(out_c + output_chans*b))];
                } // out_c
                out_input_grad[idx] = value;
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
        DevProp dev = get_thread_blocks(total_count);

        AT_DISPATCH_FLOATING_TYPES(
                guide.type(),
                "bilateral_slice_cuda_forward",
                ([&] {
                 bilateral_slice_cuda_forward_kernel<scalar_t><<<dev.blocks, dev.threads>>>(
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

std::vector<at::Tensor>
bilateral_slice_cuda_backward(at::Tensor grid_grad,
                              at::Tensor guide_grad,
                              at::Tensor input_grad,
                              at::Tensor upstream_grad,
                              at::Tensor bilateral_grid,
                              at::Tensor guide,
                              at::Tensor input,
                              GridSizes& gsz,
                              bool has_offset)
{
        int h = gsz.h;
        int w = gsz.w;
        int bs = gsz.bs;
        int coeffs_chans = gsz.coeffs_chans;
        int gd = gsz.gd;
        int gh = gsz.gh;
        int gw = gsz.gw;
        int input_chans = gsz.input_chans;

        int output_chans = 0;
        if (has_offset) {
                output_chans = coeffs_chans/(input_chans + 1);
        } else {
                output_chans = coeffs_chans/input_chans;
        }

        int grid_count = bs*gh*gw*gd*coeffs_chans;
        DevProp dev = get_thread_blocks(grid_count);
        AT_DISPATCH_FLOATING_TYPES(
                guide.type(),
                "bilateral_slice_cuda_grid_grad_kernel",
                ([&] {
                 bilateral_slice_cuda_grid_grad_kernel<scalar_t><<<dev.blocks, dev.threads>>>(
                         grid_grad.data<scalar_t>(),
                         upstream_grad.data<scalar_t>(),
                         guide.data<scalar_t>(),
                         input.data<scalar_t>(),
                         gsz,
                         has_offset,
                         grid_count,
                         output_chans);
                 }));

        int guide_count = bs*h*w;
        dev = get_thread_blocks(guide_count);
        AT_DISPATCH_FLOATING_TYPES(
                guide.type(),
                "bilateral_slice_cuda_guide_grad_kernel",
                ([&] {
                 bilateral_slice_cuda_guide_grad_kernel<scalar_t><<<dev.blocks, dev.threads>>>(
                         guide_grad.data<scalar_t>(),
                         upstream_grad.data<scalar_t>(),
                         bilateral_grid.data<scalar_t>(),
                         guide.data<scalar_t>(),
                         input.data<scalar_t>(),
                         gsz,
                         has_offset,
                         guide_count,
                         output_chans);
                 }));

        int input_count = bs*h*w*input_chans;
        dev = get_thread_blocks(input_count);
        AT_DISPATCH_FLOATING_TYPES(
                input.type(),
                "bilateral_slice_cuda_input_grad_kernel",
                ([&] {
                 bilateral_slice_cuda_input_grad_kernel<scalar_t><<<dev.blocks, dev.threads>>>(
                         input_grad.data<scalar_t>(),
                         upstream_grad.data<scalar_t>(),
                         bilateral_grid.data<scalar_t>(),
                         guide.data<scalar_t>(),
                         gsz,
                         has_offset,
                         input_count,
                         output_chans);
                 }));

        return {grid_grad, guide_grad, input_grad};
}
