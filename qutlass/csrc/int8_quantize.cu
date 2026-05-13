#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include <algorithm>
#include <cstdint>

#include "int8_quantize.h"

namespace QUTLASS {
namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 4096;

bool is_supported_dtype(at::ScalarType dtype) {
    return dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16;
}

int64_t flattened_row_count(torch::Tensor const& input) {
    if (input.dim() == 0) {
        return 1;
    }

    int64_t rows = 1;
    for (int64_t dim = 0; dim < input.dim() - 1; ++dim) {
        rows *= input.size(dim);
    }
    return rows;
}

template <typename T>
__device__ __forceinline__ float to_float(T value) {
    return static_cast<float>(value);
}

template <typename InputT, typename ScaleT>
__global__ void quantize_int8_kernel(InputT const* __restrict__ input,
                                     ScaleT const* __restrict__ scale,
                                     int8_t* __restrict__ output,
                                     int64_t numel,
                                     int64_t row_stride,
                                     bool per_tensor_scale) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < numel;
         idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int64_t const scale_idx = per_tensor_scale ? 0 : idx / row_stride;
        // Symmetric quantization with explicit regular scale: q = input / scale.
        float scaled = to_float(input[idx]) / to_float(scale[scale_idx]);
        if (scaled != scaled) {
            scaled = 0.0f;
        }

        scaled = fminf(fmaxf(scaled, -128.0f), 127.0f);
        // __float2int_rn rounds to nearest even after the explicit int8 clamp.
        output[idx] = static_cast<int8_t>(__float2int_rn(scaled));
    }
}

template <typename InputT, typename ScaleT>
void launch_quantize_int8(torch::Tensor const& input,
                          torch::Tensor const& scale,
                          torch::Tensor& output,
                          int64_t row_stride,
                          bool per_tensor_scale,
                          cudaStream_t stream) {
    int64_t const numel = input.numel();
    if (numel == 0) {
        return;
    }

    int const blocks = static_cast<int>(
        std::min<int64_t>((numel + kThreads - 1) / kThreads, kMaxBlocks));
    quantize_int8_kernel<InputT, ScaleT><<<blocks, kThreads, 0, stream>>>(
        input.data_ptr<InputT>(),
        scale.data_ptr<ScaleT>(),
        output.data_ptr<int8_t>(),
        numel,
        row_stride,
        per_tensor_scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

torch::Tensor quantize_int8_host(torch::Tensor const& input,
                                 torch::Tensor const& scale) {
    torch::checkDeviceType("quantize_int8_host", {input, scale}, at::DeviceType::CUDA);
    torch::checkAllSameGPU("quantize_int8_host",
                           {{input, "input", 0}, {scale, "scale", 1}});
    TORCH_CHECK(is_supported_dtype(input.scalar_type()),
                "input must be bf16, fp16, or fp32");
    TORCH_CHECK(is_supported_dtype(scale.scalar_type()),
                "scale must be a regular bf16, fp16, or fp32 tensor");

    torch::Tensor input_contig = input.contiguous();
    torch::Tensor scale_contig = scale.contiguous();
    int64_t const row_count = flattened_row_count(input_contig);
    bool const per_tensor_scale = scale_contig.numel() == 1;
    bool const per_row_scale = scale_contig.numel() == row_count;
    TORCH_CHECK(per_tensor_scale || per_row_scale,
                "scale must contain either 1 value or one value per flattened "
                "input row; expected 1 or ",
                row_count,
                " values, got ",
                scale_contig.numel());

    auto output = torch::empty(input_contig.sizes(),
                               input_contig.options().dtype(torch::kInt8));
    if (input_contig.numel() == 0) {
        return output;
    }

    int64_t const row_stride = input_contig.dim() == 0 ? 1 : input_contig.size(-1);
    TORCH_CHECK(row_stride > 0,
                "input last dimension must be non-zero for non-empty input");

    c10::cuda::CUDAGuard device_guard(input_contig.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(input_contig.get_device());

#define QUTLASS_LAUNCH_INT8_QUANTIZE(INPUT_T)                                      \
    do {                                                                           \
        if (scale_contig.scalar_type() == at::kFloat) {                            \
            launch_quantize_int8<INPUT_T, float>(input_contig,                     \
                                                 scale_contig,                     \
                                                 output,                           \
                                                 row_stride,                       \
                                                 per_tensor_scale,                 \
                                                 stream);                          \
        } else if (scale_contig.scalar_type() == at::kHalf) {                      \
            launch_quantize_int8<INPUT_T, c10::Half>(input_contig,                 \
                                                     scale_contig,                 \
                                                     output,                       \
                                                     row_stride,                   \
                                                     per_tensor_scale,             \
                                                     stream);                      \
        } else {                                                                   \
            launch_quantize_int8<INPUT_T, c10::BFloat16>(input_contig,             \
                                                         scale_contig,             \
                                                         output,                   \
                                                         row_stride,               \
                                                         per_tensor_scale,         \
                                                         stream);                  \
        }                                                                          \
    } while (false)

    if (input_contig.scalar_type() == at::kFloat) {
        QUTLASS_LAUNCH_INT8_QUANTIZE(float);
    } else if (input_contig.scalar_type() == at::kHalf) {
        QUTLASS_LAUNCH_INT8_QUANTIZE(c10::Half);
    } else {
        QUTLASS_LAUNCH_INT8_QUANTIZE(c10::BFloat16);
    }

#undef QUTLASS_LAUNCH_INT8_QUANTIZE

    return output;
}

}  // namespace QUTLASS
