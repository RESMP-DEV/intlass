#pragma once

#include <torch/types.h>

namespace QUTLASS {

torch::Tensor quantize_int8_host(torch::Tensor const& input,
                                 torch::Tensor const& scale);

}  // namespace QUTLASS
