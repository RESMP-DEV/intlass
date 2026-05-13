#pragma once

#include <torch/types.h>

namespace QUTLASS {

torch::Tensor pack_int4_host(torch::Tensor const& input);
torch::Tensor quantize_int8_host(torch::Tensor const& input, torch::Tensor const& scale);
void matmul_host_int4_bf16_tn(torch::Tensor& out, torch::Tensor const& A, torch::Tensor const& B, torch::Tensor const& A_scale, torch::Tensor const& B_scale);
void matmul_host_int8_bf16_tn(torch::Tensor& out, torch::Tensor const& A, torch::Tensor const& B, torch::Tensor const& A_scale, torch::Tensor const& B_scale);

}  // namespace QUTLASS
