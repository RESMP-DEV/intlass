#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "int_kernels.h"

namespace QUTLASS {
namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 4096;

__global__ void pack_int4_kernel(int8_t const* __restrict__ input,
                                 uint8_t* __restrict__ output,
                                 int64_t num_output_bytes) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < num_output_bytes;
         idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        uint8_t const even = static_cast<uint8_t>(input[idx * 2]) & 0x0Fu;
        uint8_t const odd = static_cast<uint8_t>(input[idx * 2 + 1]) & 0x0Fu;
        // low nibble stores even element; high nibble stores odd element
        output[idx] = static_cast<uint8_t>(even | (odd << 4));
    }
}

}  // namespace

torch::Tensor pack_int4_host(torch::Tensor const& input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kInt8, "input must be torch.int8");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() > 0, "input must have non-empty rank");
    TORCH_CHECK(input.size(-1) % 2 == 0, "input last dimension must be even");

    std::vector<int64_t> output_sizes(input.sizes().begin(), input.sizes().end());
    output_sizes.back() /= 2;
    auto output = torch::empty(output_sizes, input.options().dtype(torch::kUInt8));

    int64_t const num_output_bytes = output.numel();
    if (num_output_bytes == 0) {
        return output;
    }

    c10::cuda::CUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());

    int const blocks = static_cast<int>(
        std::min<int64_t>((num_output_bytes + kThreads - 1) / kThreads, kMaxBlocks));
    pack_int4_kernel<<<blocks, kThreads, 0, stream>>>(
        input.data_ptr<int8_t>(),
        output.data_ptr<uint8_t>(),
        num_output_bytes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}

}  // namespace QUTLASS
