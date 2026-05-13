# Changelog

## [Unreleased]

### Added

- Added an SM86-gated packed signed INT4 x INT4 to BF16 GEMM host path with
  explicit regular scales for `matmul_host_int4_bf16_tn`.
- Added an SM86-gated INT8 x INT8 to BF16 GEMM host path with explicit regular
  scales for `matmul_host_int8_bf16_tn`.
- Added a CUDA int8 quantization launcher and Python wrapper that use explicit
  per-tensor or per-row regular floating-point scales.
- Added `LEAN-CTX.md` as the compact agent contract for Ampere INT4/INT8 migration work, including source-first guidance for local SM86 INT4, MXFP4, FP8, and Marlin evidence.
