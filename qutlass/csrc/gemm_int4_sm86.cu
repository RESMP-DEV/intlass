#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include <cstdint>

#include "int_kernels.h"

namespace QUTLASS {
namespace {

constexpr int kTileRows = 16;
constexpr int kTileCols = 16;

bool is_regular_scale_type(at::ScalarType dtype) {
    return dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16;
}

void check_scale(char const* name, torch::Tensor const& scale, int64_t rows) {
    TORCH_CHECK(scale.defined(), name, " must be defined");
    TORCH_CHECK(scale.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(scale.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(scale.numel() > 0, name, " must contain at least one explicit scale value");
    TORCH_CHECK(is_regular_scale_type(scale.scalar_type()),
                name,
                " must be a regular fp32, fp16, or bf16 scale tensor");
    TORCH_CHECK(scale.numel() == 1 || scale.numel() == rows,
                name,
                " must contain either 1 value or one value per row; expected 1 or ",
                rows,
                " values, got ",
                scale.numel());
}

void check_sm86(torch::Tensor const& tensor) {
    c10::cuda::CUDAGuard device_guard(tensor.device());
    cudaDeviceProp props;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&props, tensor.get_device()));
    TORCH_CHECK(props.major == 8 && props.minor == 6,
                "matmul_host_int4_bf16_tn requires an SM86 CUDA device, got sm_",
                props.major,
                props.minor);
}

template <typename T>
__device__ __forceinline__ float scale_to_float(T value) {
    return static_cast<float>(value);
}

__device__ __forceinline__ int32_t unpack_signed_int4(uint8_t value, bool high_nibble) {
    uint8_t const nibble = high_nibble ? ((value >> 4) & 0x0Fu) : (value & 0x0Fu);
    return static_cast<int32_t>((nibble & 0x08u) ? static_cast<int8_t>(nibble | 0xF0u)
                                                : static_cast<int8_t>(nibble));
}

// Source lineage: contrib/int4_kernel/kernel/int4_gemm_sm86.cu documents the
// packed signed INT4, LSB-first nibble contract this correctness path follows.
// Future Ampere Tensor Core path evidence:
// mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32
template <typename AScaleT, typename BScaleT>
__global__ void matmul_int4_bf16_tn_kernel(uint8_t const* __restrict__ A,
                                           uint8_t const* __restrict__ B,
                                           AScaleT const* __restrict__ A_scale,
                                           BScaleT const* __restrict__ B_scale,
                                           __nv_bfloat16* __restrict__ out,
                                           int64_t M,
                                           int64_t N,
                                           int64_t K_packed,
                                           bool a_per_tensor_scale,
                                           bool b_per_tensor_scale) {
    int64_t const m = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    int64_t const n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (m >= M || n >= N) {
        return;
    }

    int32_t acc = 0;
    for (int64_t k_byte = 0; k_byte < K_packed; ++k_byte) {
        uint8_t const a_byte = A[m * K_packed + k_byte];
        uint8_t const b_byte = B[n * K_packed + k_byte];
        acc += unpack_signed_int4(a_byte, false) * unpack_signed_int4(b_byte, false);
        acc += unpack_signed_int4(a_byte, true) * unpack_signed_int4(b_byte, true);
    }

    float const a_scale = scale_to_float(A_scale[a_per_tensor_scale ? 0 : m]);
    float const b_scale = scale_to_float(B_scale[b_per_tensor_scale ? 0 : n]);
    float const value = static_cast<float>(acc) * a_scale * b_scale;
    out[m * N + n] = __float2bfloat16_rn(value);
}

template <typename AScaleT, typename BScaleT>
void launch_matmul_int4_bf16_tn(torch::Tensor& out,
                                torch::Tensor const& A,
                                torch::Tensor const& B,
                                torch::Tensor const& A_scale,
                                torch::Tensor const& B_scale,
                                cudaStream_t stream) {
    int64_t const M = A.size(0);
    int64_t const N = B.size(0);
    int64_t const K_packed = A.size(1);
    if (M == 0 || N == 0) {
        return;
    }

    dim3 const block(kTileCols, kTileRows);
    dim3 const grid(static_cast<unsigned int>((N + kTileCols - 1) / kTileCols),
                    static_cast<unsigned int>((M + kTileRows - 1) / kTileRows));

    matmul_int4_bf16_tn_kernel<AScaleT, BScaleT><<<grid, block, 0, stream>>>(
        A.data_ptr<uint8_t>(),
        B.data_ptr<uint8_t>(),
        A_scale.data_ptr<AScaleT>(),
        B_scale.data_ptr<BScaleT>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<c10::BFloat16>()),
        M,
        N,
        K_packed,
        A_scale.numel() == 1,
        B_scale.numel() == 1);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename AScaleT>
void dispatch_b_scale(torch::Tensor& out,
                      torch::Tensor const& A,
                      torch::Tensor const& B,
                      torch::Tensor const& A_scale,
                      torch::Tensor const& B_scale,
                      cudaStream_t stream) {
    if (B_scale.scalar_type() == at::kFloat) {
        launch_matmul_int4_bf16_tn<AScaleT, float>(out, A, B, A_scale, B_scale, stream);
    } else if (B_scale.scalar_type() == at::kHalf) {
        launch_matmul_int4_bf16_tn<AScaleT, c10::Half>(out, A, B, A_scale, B_scale, stream);
    } else {
        launch_matmul_int4_bf16_tn<AScaleT, c10::BFloat16>(out, A, B, A_scale, B_scale, stream);
    }
}

void dispatch_scales(torch::Tensor& out,
                     torch::Tensor const& A,
                     torch::Tensor const& B,
                     torch::Tensor const& A_scale,
                     torch::Tensor const& B_scale,
                     cudaStream_t stream) {
    if (A_scale.scalar_type() == at::kFloat) {
        dispatch_b_scale<float>(out, A, B, A_scale, B_scale, stream);
    } else if (A_scale.scalar_type() == at::kHalf) {
        dispatch_b_scale<c10::Half>(out, A, B, A_scale, B_scale, stream);
    } else {
        dispatch_b_scale<c10::BFloat16>(out, A, B, A_scale, B_scale, stream);
    }
}

}  // namespace

void matmul_host_int4_bf16_tn(torch::Tensor& out,
                              torch::Tensor const& A,
                              torch::Tensor const& B,
                              torch::Tensor const& A_scale,
                              torch::Tensor const& B_scale) {
    TORCH_CHECK(out.defined(), "out must be defined");
    TORCH_CHECK(A.defined() && B.defined(), "A and B must be defined");
    TORCH_CHECK(A_scale.defined() && B_scale.defined(),
                "A_scale and B_scale must be defined");
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && out.is_cuda() &&
                    A_scale.is_cuda() && B_scale.is_cuda(),
                "A, B, out, A_scale, and B_scale must be CUDA tensors");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && out.is_contiguous(),
                "A, B, and out must be contiguous");
    TORCH_CHECK(A.scalar_type() == at::kByte, "A must be packed torch.uint8");
    TORCH_CHECK(B.scalar_type() == at::kByte, "B must be packed torch.uint8");
    TORCH_CHECK(out.scalar_type() == at::kBFloat16, "out must be torch.bfloat16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D packed INT4 tensors");
    TORCH_CHECK(A.size(1) == B.size(1),
                "Packed K-byte dimensions must match for A @ B.T");
    TORCH_CHECK(out.dim() == 2 && out.size(0) == A.size(0) && out.size(1) == B.size(0),
                "out must have shape (A.size(0), B.size(0))");

    check_scale("A_scale", A_scale, A.size(0));
    check_scale("B_scale", B_scale, B.size(0));
    torch::checkAllSameGPU("matmul_host_int4_bf16_tn",
                           {{out, "out", 0},
                            {A, "A", 1},
                            {B, "B", 2},
                            {A_scale, "A_scale", 3},
                            {B_scale, "B_scale", 4}});
    check_sm86(A);

    c10::cuda::CUDAGuard device_guard(A.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());
    dispatch_scales(out, A, B, A_scale, B_scale, stream);
}

}  // namespace QUTLASS
