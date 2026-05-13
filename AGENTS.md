# Repository Guidelines

Read `LEAN-CTX.md` before starting implementation tasks. It is the compact operating contract for Ampere INT4/INT8 work and lists the local source evidence agents should inspect before broad repository searches.

## Project Mission

This checkout is being converted from Blackwell-focused QuTLASS code into an Ampere-focused INT4/INT8 CUDA extension. Treat existing FP4, FP8, MXFP4, MXFP8, and NVFP4 paths as migration sources, not the target contract. Ampere MXFP4/FP8 evidence already exists in nearby local projects, so agents should cite and adapt that evidence before writing new kernels. New active work should replace microscaling/float8 assumptions with explicit INT4/INT8 packing, scale, zero-point, and accumulation behavior for Ampere Tensor Cores (`sm_80`/`sm_86` first).

## Project Structure & Module Organization

The current Python package surface is in `qutlass/`, with wrappers in `qutlass/__init__.py` and tensor layout helpers in `qutlass/utils.py`. Native extension bindings live in `qutlass/csrc/bindings.cpp`; CUDA kernels and host launchers live under `qutlass/csrc/`; shared CUTLASS extension headers are under `qutlass/csrc/include/`. Correctness tests are in `tests/`, benchmarks are in `benchmarks/`, and figures are in `assets/`. `third_party/cutlass` is a Git submodule and should stay submodule-managed.

## Build, Test, and Development Commands

```bash
pip install -r requirements.txt
git submodule update --init --recursive
TORCH_CUDA_ARCH_LIST="8.0;8.6" pip install --no-build-isolation -e .
python -m pytest tests -q
python tests/mxfp4_test.py
python benchmarks/bench_mxfp4_sm120.py
```

The targeted install command documents the intended Ampere build, but `setup.py` still contains Blackwell arch flags until migrated. Use existing MXFP/NVFP tests and benchmarks as baselines while replacing them with INT4/INT8 equivalents.

## Coding Style & Naming Conventions

Use 4-space Python indentation, snake_case function names, and explicit dtype/device checks. New public names should say `int4` or `int8`, not `fp4`, `fp8`, `mxfp`, or `nvfp`, unless they are compatibility shims. Keep CUDA/C++ code C++17-compatible and name kernels by operation, datatype, layout, and architecture when useful, for example `matmul_int4_bf16_tn_sm80`.

## Testing Guidelines

Add or rename tests beside the migrated format, for example `tests/int4_test.py` and `tests/int8_test.py`. Use deterministic seeds, `torch.testing.assert_close`, and small Ampere-friendly reference problems before large LLM shapes. Tests that require CUDA should skip clearly when CUDA is unavailable, but extension changes are not validated by CPU-only import checks.

## Commit & Pull Request Guidelines

Recent history uses short imperative subjects such as `add fp8 sm120` and `remove flashinfer dependency`; keep that concise style while naming the migrated lane, for example `add int4 sm80 matmul`. PRs should list the affected API names, CUDA architecture tested, GPU model, build command, correctness tests, and benchmark command.

## Agent-Specific Notes

When converting a path, update the Python wrapper, binding checks, kernel launcher, tests, and benchmark labels in the same pass. Do not leave Blackwell-only scale dtypes or `sm_100`/`sm_120` assumptions on an Ampere INT path.

Keep task context narrow. Start from the task prompt, `LEAN-CTX.md`, and the named source files; do not load the full repository just to make a local change.

AlphaHENG tasks scoped as `scope: contrib/intlass/` execute from this directory, not from the AlphaHENG root. In those tasks, use scoped-relative paths everywhere: `setup.py`, `qutlass/csrc/bindings.cpp`, `tests/int4_pack_test.py`. Do not write `contrib/intlass/setup.py` in prompts, `change_paths`, `verify_command`, or diff metadata; that creates doubled paths under the worker worktree and fails validation.
