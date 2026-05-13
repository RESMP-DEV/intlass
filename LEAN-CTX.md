# Lean Agent Context

Use this file as the first-pass contract for `contrib/intlass` tasks. Do not scan the full repository before acting. Read only the files named in the task prompt, then expand to the source list below only when needed.

## Mission

Convert this checkout from Blackwell QuTLASS FP4/FP8/MXFP/NVFP work into an Ampere `sm_86` INT4/INT8 CUDA extension. The final active API should use explicit INT4 and INT8 contracts, not Blackwell `sm_100`/`sm_120` assumptions.

## Source-First Rule

Do not implement Ampere kernels from scratch until existing evidence has been checked. Use these local sources first:

- `../int4_kernel/kernel/int4_gemm_sm86.cu`
- `../int4_kernel/kernel/int4_gemm_persistent_sm86.cu`
- `../int4_kernel/kernel/w4a8_gemm_sm86.cu`
- `../int4_kernel/kernel/int4_pack_sm86.h`
- `../ktransformers/kt-kernel/cuda/mxfp4/mxfp4_dequant.cu`
- `../ktransformers/kt-kernel/cuda/fp8/fp8_linear.cu`
- `../ktransformers/kt-kernel/cuda/gptq_marlin/`
- `../ktransformers/kt-kernel/scripts/moe_reap_calibration_multi_gpu.py`
- `../ktransformers/kt-kernel/scripts/README.md`

Treat MXFP4 and FP8 paths as evidence for packing, scale layout, model artifact contracts, and fallback conversion strategy. The target for this repo remains Ampere INT4/INT8 unless the task explicitly asks for an evidence report.

## Task Discipline

Each task should touch only its requested files. If a dependency is missing, write a focused artifact explaining the blocker instead of broadening scope. Prefer source citations, symbol names, and exact build/test commands over general design text.

## Architecture Guardrails

Active CUDA build flags should target `compute_86,code=sm_86`. Avoid `compute_100a`, `compute_120a`, `sm_100`, and `sm_120` in active Ampere code paths. Use Ampere INT MMA evidence such as `mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32` and `mma.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32` when implementing INT kernels.
