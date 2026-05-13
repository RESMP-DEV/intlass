# IntLASS: Ampere SM86 INT4/INT8 CUDA Extension
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Static Badge](https://img.shields.io/badge/PyTorch-2.8-red)](https://download.pytorch.org/whl/nightly/cu128)

**Active project:** Ampere `sm_86` INT4/INT8 packing, quantization, and BF16-output GEMM for NVIDIA Ampere Tensor Cores.

This checkout is being migrated from the original Blackwell QuTLASS MXFP/NVFP codebase into an Ampere-focused CUDA extension. New work should target explicit integer quantization contracts: packed INT4 data, INT8 data, scale and zero-point handling, and integer accumulation paths feeding BF16 results.

Primary active API surface:

- `pack_int4`
- `quantize_int8`
- `matmul_int4_bf16_tn`
- `matmul_int8_bf16_tn`

The intended active architecture target is `sm_86` first, with `sm_80` compatibility where practical. MXFP, NVFP, FP4, FP8, and Blackwell-specific CUTLASS paths are retained only as legacy/source material unless a task explicitly asks for them.

---

## Table of Contents
- [Active Ampere Scope](#active-ampere-scope)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Validation](#validation)
- [Legacy/Source Material: MXFP, NVFP, and Blackwell](#legacysource-material-mxfp-nvfp-and-blackwell)
- [Citation](#citation)

---

## Active Ampere Scope

The active implementation direction is an INT4/INT8 CUDA extension for Ampere:

- INT4 packing via `pack_int4`
- INT8 quantization via `quantize_int8`
- BF16-output Tensor Core GEMM via `matmul_int4_bf16_tn`
- BF16-output Tensor Core GEMM via `matmul_int8_bf16_tn`
- Explicit scale, zero-point, layout, and accumulation behavior for Ampere kernels

Ampere kernel work should avoid Blackwell-only assumptions such as `sm_100`, `sm_120`, MXFP block-scale layouts as the public contract, or FP8 scale dtypes as the active interface.

## Getting Started

### Requirements

- NVIDIA Ampere GPU, with `sm_86` as the primary target
- CUDA 12.x toolchain compatible with the local PyTorch build
- PyTorch extension build environment

### Installation

Install Python requirements:

```bash
pip install -r requirements.txt
```

Install the extension in editable mode for Ampere:

```bash
TORCH_CUDA_ARCH_LIST="8.0;8.6" pip install --no-build-isolation -e .
```

## Usage

The active public names are expected to be:

```python
import qutlass

packed_w = qutlass.pack_int4(w_int)
acts_i8, act_scale, act_zero = qutlass.quantize_int8(acts)

out_i4 = qutlass.matmul_int4_bf16_tn(acts_bf16, packed_w, scales, zeros)
out_i8 = qutlass.matmul_int8_bf16_tn(a_int8, b_int8, a_scale, b_scale)
```

Exact argument order and tensor layout requirements should be kept in sync with the bindings and tests as the migration lands.

## Validation

Use focused INT tests and Ampere build flags while this migration is in progress:

```bash
TORCH_CUDA_ARCH_LIST="8.0;8.6" pip install --no-build-isolation -e .
python -m pytest tests -q
```

CUDA-dependent tests should skip clearly when CUDA is unavailable. CPU-only import checks do not validate extension behavior.

## Legacy/Source Material: MXFP, NVFP, and Blackwell

The original QuTLASS README content below describes the Blackwell-focused MXFP/NVFP project that this checkout is being migrated away from. Treat this section as historical reference and source material for migration only. It is not the active project contract for new Ampere INT4/INT8 work.

### Original QuTLASS v0.2 Identity

QuTLASS was a CUTLASS-powered quantized BLAS library for low-bit deep learning on NVIDIA Blackwell GPUs.

It introduced narrow-precision microscaling routines tailored for quantized LLM inference and training on NVIDIA Blackwell GPUs.

[![arXiv](https://img.shields.io/badge/arXiv-2509.23202-b31b1b.svg)](https://arxiv.org/pdf/2509.23202)
[![arXiv](https://img.shields.io/badge/arXiv-2505.14669-b31b1b.svg)](https://arxiv.org/abs/2505.14669)

### Microscaling in Blackwell

The Blackwell architecture supports native matrix multiplication with microscaling, using scale factors in the form:

$$
D = C + (A \times \mathrm{SFA}) \cdot (B \times \mathrm{SFB})
$$

The scale factors are applied along the inner ($K$) dimension of the GEMM. For MXFP types, one scale factor is shared by every 32 elements along $K$ (`gs=32`). For an $M \times K$ matrix $A$, the corresponding scale matrix $\mathrm{SFA}$ has dimensions:

$$
M \times \left\lceil K / gs \right\rceil
$$

### Legacy QuTLASS v0.2 Features

- FlashInfer backend support for B200 GPUs
- Quantization-aware training via MXFP types
- Quartet clipping mask computation integrated in quantization routines
- Prototype backward kernels for MXFP4 (`sm_120`) and MXFP8 (`sm_100`)
- CUTLASS MXFP8 backward GEMM kernels in TN and NN layouts
- Transformers QAT integration
- Nanochat-QAT integration

### Legacy QuTLASS v0.1 Features

- Support for `sm_100` GPUs, including NVIDIA B200
- NVFP4 microscaling with W4A4 quantization support
- Online rotations with fused transform, quantization, and scale computation
- Runtime-loaded rotation matrices
- CUTLASS-backed NVFP4:NVFP4 matmul with block-scale reordering
- Abs-max quantization
- Multiple rotation sizes for MXFP4 and NVFP4
- vLLM integration

### Legacy QuTLASS v0.0 Features

- MXFP4 microscaling support
- Weight and activation quantization (`W4A4`)
- Online transforms, quantization, and scale computation
- Microscaling group-size-compatible transformations
- Quartet and abs-max quantization schemes
- CUTLASS-backed MXFP4:MXFP4 matmul with block-scale reordering
- Small-batch prototype MXFP4 kernel without reordering
- Transformers integration

### Legacy Usage Notes

Legacy MXFP4 correctness tests were run with:

```bash
python tests/mxfp4_test.py
```

Legacy MXFP4 benchmarks were run with:

```bash
python benchmarks/bench_mxfp4.py
```

The legacy fused quantization kernel was exposed as `qutlass.fusedQuantizeMx(a, h, method)`, where `method` was `Literal["quest", "abs_max"]`. It returned FP4 (`e2m1`) quantized data and FP8 (`e8m0`) scaling factors.

The legacy matmul path used `qutlass.matmul_mxf4_bf16_tn(aq, bq, a_sf, b_sf, alpha)` with scale factors rearranged by `qutlass.to_blocked` into the cuBLAS block-scaled swizzle format. A custom prototype path, `qutlass.matmul_ada_mxf4_bf16_tn(...)`, avoided that reordering for small batch sizes. NVFP4 was treated as functionally equivalent aside from naming.

### Legacy Benchmark Material

The original benchmark figures and claims measured MXFP4/NVFP4 behavior on Blackwell and RTX 5090-class targets. Keep these assets as historical references while replacing benchmark labels and commands with INT4/INT8 equivalents.

In the original README, microbenchmarks showed MXFP4 performance across batch sizes, including ideal matrix multiplication and full-pipeline measurements with Hadamard rotation, data quantization, scale computation, and block-scale reordering.

Original end-to-end inference notes described MXFP4 speedups over PyTorch BF16 in Transformers for 8B and 14B models. Original training notes described MXFP4:MXFP8 QAT on Llama-3.1-8B.

For historical recipes related to MXFP and NVFP formats, the original project referenced [FP-Quant](https://github.com/IST-DASLab/FP-Quant), [nanochat-qat](https://github.com/IST-DASLab/nanochat-qat/pull/1), and related Transformers integrations.

## Citation

```bibtex
@misc{qutlass2025,
      title={QuTLASS: CUTLASS-Powered Quantized BLAS for Deep Learning},
      author={Roberto L. Castro, and Dan Alistarh},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/IST-DASLab/qutlass}},
}
```
