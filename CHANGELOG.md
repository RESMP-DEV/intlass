# Changelog

## [Unreleased]

### Added

- Added deterministic signed INT4 pack/unpack tests, including an SM86-gated
  CUDA extension comparison for `qutlass._CUDA.pack_int4`.
- Added deterministic SM86-gated `matmul_int4_bf16_tn` correctness tests
  against pure-Python packed signed INT4 inputs and explicit-scale BF16
  PyTorch references.
- Added deterministic SM86-gated `matmul_int8_bf16_tn` correctness tests
  against explicit-scale BF16 PyTorch references.
- Added an SM86-only INT8 benchmark that reports the BF16 PyTorch baseline and
  `matmul_int8_bf16_tn` timing.
- Added an SM86-only INT4 benchmark that reports the BF16 PyTorch baseline and
  `matmul_int4_bf16_tn` timing.
- Added an SM86-gated packed signed INT4 x INT4 to BF16 GEMM host path with
  explicit regular scales for `matmul_host_int4_bf16_tn`.
- Added an SM86-gated INT8 x INT8 to BF16 GEMM host path with explicit regular
  scales for `matmul_host_int8_bf16_tn`.
- Added a CUDA int8 quantization launcher and Python wrapper that use explicit
  per-tensor or per-row regular floating-point scales.
- Added deterministic INT8 quantization tests for explicit per-tensor and
  per-row scales, including an SM86-gated CUDA extension comparison.
- Added `LEAN-CTX.md` as the compact agent contract for Ampere INT4/INT8 migration work, including source-first guidance for local SM86 INT4, MXFP4, FP8, and Marlin evidence.
