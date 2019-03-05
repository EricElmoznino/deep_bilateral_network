#include "bilateral_slice.h"
#include "torch/extension.h"

#include <vector>

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DIM(x, n) AT_CHECK(x.ndimension() == n, #x " must be " #n "D")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor
bilateral_slice_cuda_forward(at::Tensor output_tensor,
                             at::Tensor bilateral_grid,
                             at::Tensor guide,
                             at::Tensor input,
                             GridSizes& gsz,
                             bool has_offset);

at::Tensor
bilateral_slice_cuda_backward(at::Tensor grid_grad,
                              at::Tensor guide_grad,
                              at::Tensor input_grad,
                              at::Tensor upstream_grad,
                              at::Tensor bilateral_grid,
                              at::Tensor guide,
                              at::Tensor input,
                              GridSizes& gsz,
                              bool has_offset);

at::Tensor
bilateral_slice_forward(at::Tensor bilateral_grid,
                        at::Tensor guide,
                        at::Tensor input,
                        bool has_offset)
{
        CHECK_INPUT(bilateral_grid);
        CHECK_INPUT(guide);
        CHECK_INPUT(input);

        CHECK_DIM(bilateral_grid, 5);
        CHECK_DIM(guide, 3);
        CHECK_DIM(input, 4);

        int64_t h = guide.sizes()[1];
        int64_t w = guide.sizes()[2];
        int64_t bs = bilateral_grid.sizes()[0];
        int64_t coeffs_chans = bilateral_grid.sizes()[1];
        int64_t gd = bilateral_grid.sizes()[2];
        int64_t gh = bilateral_grid.sizes()[3];
        int64_t gw = bilateral_grid.sizes()[4];
        int64_t input_chans = input.sizes()[1];
        GridSizes grid_sizes{.h = h,
                             .w = w,
                             .bs = bs,
                             .coeffs_chans = coeffs_chans,
                             .gd = gd,
                             .gh = gh,
                             .gw = gw,
                             .input_chans = input_chans};

        AT_CHECK((input.sizes()[0] == guide.sizes()[0]) &&
                 (input.sizes()[2] == h) &&
                 (input.sizes()[3] == w),
                 "Input and guide size should match");
        AT_CHECK(guide.sizes()[0] == bs, "Batch sizes should match.");

        int64_t output_chans;
        if (has_offset) {
                AT_CHECK((coeffs_chans % (input_chans + 1)) == 0,
                         "Slicing with affine offset, coefficients grid "
                         "should have n_out*(n_in+1) channels.");
                output_chans = coeffs_chans/(input_chans + 1);
        } else {
                AT_CHECK((coeffs_chans % input_chans) == 0,
                         "Slicing without affine offset, coefficients grid "
                         "should have n_out*n_in channels.");
                output_chans = coeffs_chans / input_chans;
        }

        at::Tensor output_tensor = at::empty({bs, output_chans, h, w},
                                             input.options());

        return bilateral_slice_cuda_forward(output_tensor,
                                            bilateral_grid,
                                            guide,
                                            input,
                                            grid_sizes,
                                            has_offset);
}

at::Tensor
bilateral_slice_backward(at::Tensor upstream_grad,
                         at::Tensor bilateral_grid,
                         at::Tensor guide,
                         at::Tensor input,
                         bool has_offset)
{
        CHECK_INPUT(upstream_grad);
        CHECK_INPUT(bilateral_grid);
        CHECK_INPUT(guide);
        CHECK_INPUT(input);

        CHECK_DIM(upstream_grad, 4);
        CHECK_DIM(bilateral_grid, 5);
        CHECK_DIM(guide, 3);
        CHECK_DIM(input, 4);

        at::Tensor grid_grad = at::empty(bilateral_grid.sizes(),
                                         bilateral_grid.options());
        at::Tensor guide_grad = at::empty(guide.sizes(),
                                          guide.options());
        at::Tensor input_grad = at::empty(input.sizes(),
                                          input.options());

        GridSizes grid_sizes{.h = guide.sizes()[1],
                             .w = guide.sizes()[2],
                             .bs = bilateral_grid.sizes()[0],
                             .coeffs_chans = bilateral_grid.sizes()[1],
                             .gd = bilateral_grid.sizes()[2],
                             .gh = bilateral_grid.sizes()[3],
                             .gw = bilateral_grid.sizes()[4],
                             .input_chans = input.sizes()[1]};

        return bilateral_slice_cuda_backward(grid_grad,
                                             guide_grad,
                                             input_grad,
                                             upstream_grad,
                                             bilateral_grid,
                                             guide,
                                             input,
                                             grid_sizes,
                                             has_offset);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("forward", &bilateral_slice_forward, "Bilateral slice forward (CUDA)");
        m.def("backward", &bilateral_slice_backward, "Bilateral slice backward (CUDA)");
}
