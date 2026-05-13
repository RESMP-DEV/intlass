#!/usr/bin/env python3
"""Static contract checks for the Ampere INT4/INT8 IntLASS migration."""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_APIS = (
    "pack_int4",
    "quantize_int8",
    "matmul_int4_bf16_tn",
    "matmul_int8_bf16_tn",
)
INT_SOURCES = (
    "qutlass/csrc/int4_pack.cu",
    "qutlass/csrc/int8_quantize.cu",
    "qutlass/csrc/gemm_int4_sm86.cu",
    "qutlass/csrc/gemm_int8_sm86.cu",
)
TEST_CONTRACT = {
    "tests/int4_pack_test.py": ("pack_int4", "torch.testing.assert_close"),
    "tests/int8_quantize_test.py": ("quantize_int8", "torch.testing.assert_close"),
    "tests/int4_gemm_sm86_test.py": (
        "matmul_int4_bf16_tn",
        "torch.testing.assert_close",
        "8, 6",
    ),
    "tests/int8_gemm_sm86_test.py": (
        "matmul_int8_bf16_tn",
        "torch.testing.assert_close",
        "8, 6",
    ),
}
BENCHMARK_CONTRACT = {
    "benchmarks/bench_int4_sm86.py": ("matmul_int4_bf16_tn", "SM86", "require_sm86"),
    "benchmarks/bench_int8_sm86.py": ("matmul_int8_bf16_tn", "SM86", "require_sm86"),
}
BLACKWELL_ACTIVE_RE = re.compile(r"(compute_1(?:00|20)a?|sm_?1(?:00|20))")


@dataclass(frozen=True)
class Check:
    name: str
    description: str
    run: Callable[[], "Result"]


@dataclass(frozen=True)
class Result:
    ok: bool
    detail: str


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def missing_tokens(text: str, tokens: Iterable[str]) -> list[str]:
    return [token for token in tokens if token not in text]


def python_string_literals(path: Path) -> list[str]:
    text = read_text(path)
    if not text:
        return []
    try:
        tree = ast.parse(text, filename=rel(path))
    except SyntaxError:
        return []
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def pass_result(detail: str) -> Result:
    return Result(True, detail)


def fail_result(detail: str) -> Result:
    return Result(False, detail)


def check_setup_sm86_flags() -> Result:
    setup_path = ROOT / "setup.py"
    literals = python_string_literals(setup_path)
    flag = "arch=compute_86,code=sm_86"
    if flag not in literals:
        return fail_result(f"{rel(setup_path)} is missing active NVCC flag {flag!r}")
    return pass_result(f"{rel(setup_path)} declares {flag}")


def check_no_active_blackwell_flags() -> Result:
    setup_path = ROOT / "setup.py"
    offenders = sorted(
        {
            literal
            for literal in python_string_literals(setup_path)
            if BLACKWELL_ACTIVE_RE.search(literal)
        }
    )
    if offenders:
        joined = ", ".join(repr(offender) for offender in offenders)
        return fail_result(f"{rel(setup_path)} has active Blackwell arch markers: {joined}")
    return pass_result(f"{rel(setup_path)} has no active SM100/SM120 setup markers")


def check_int_source_entries() -> Result:
    setup_path = ROOT / "setup.py"
    text = read_text(setup_path)
    missing = missing_tokens(text, INT_SOURCES)
    missing_files = [source for source in INT_SOURCES if not (ROOT / source).is_file()]
    problems = []
    if missing:
        problems.append("missing setup sources: " + ", ".join(missing))
    if missing_files:
        problems.append("missing source files: " + ", ".join(missing_files))
    if problems:
        return fail_result("; ".join(problems))
    return pass_result("INT4/INT8 CUDA sources are listed and present")


def check_bindings() -> Result:
    path = ROOT / "qutlass/csrc/bindings.cpp"
    text = read_text(path)
    problems = []
    if not text:
        return fail_result(f"{rel(path)} is missing")
    for api in ACTIVE_APIS:
        api_problems = []
        if f"torch::Tensor {api}(" not in text:
            api_problems.append("wrapper")
        if text.count(f'm.def("{api}') < 2:
            api_problems.append("TORCH/PYBIND def")
        if f'm.impl("{api}"' not in text:
            api_problems.append("TORCH impl")
        if api_problems:
            problems.append(f"{api}: missing {', '.join(api_problems)}")
    if problems:
        return fail_result("; ".join(problems))
    return pass_result("bindings expose all active INT APIs through wrappers, TORCH_LIBRARY, and pybind")


def check_python_wrappers() -> Result:
    path = ROOT / "qutlass/__init__.py"
    text = read_text(path)
    if not text:
        return fail_result(f"{rel(path)} is missing")
    problems = []
    for api in ACTIVE_APIS:
        missing = missing_tokens(text, (f"def {api}(", f"qutlass._CUDA.{api}"))
        if missing:
            problems.append(f"{api}: missing " + ", ".join(missing))
    if problems:
        return fail_result("; ".join(problems))
    return pass_result("Python wrappers forward all active INT APIs to qutlass._CUDA")


def check_readme_identity() -> Result:
    path = ROOT / "README.md"
    text = read_text(path)
    if not text:
        return fail_result(f"{rel(path)} is missing")
    active_text = text.split("## Legacy/Source Material", maxsplit=1)[0]
    required = (
        "# IntLASS",
        "Ampere",
        "sm_86",
        "INT4",
        "INT8",
        "Active project",
        *ACTIVE_APIS,
    )
    missing = missing_tokens(active_text, required)
    if missing:
        return fail_result(f"{rel(path)} active identity is missing: {', '.join(missing)}")
    return pass_result("README active identity is IntLASS Ampere SM86 INT4/INT8")


def check_tests() -> Result:
    problems = []
    for filename, tokens in TEST_CONTRACT.items():
        path = ROOT / filename
        text = read_text(path)
        if not text:
            problems.append(f"{filename}: missing")
            continue
        missing = missing_tokens(text, tokens)
        if missing:
            problems.append(f"{filename}: missing " + ", ".join(missing))
    if problems:
        return fail_result("; ".join(problems))
    return pass_result("INT4/INT8 pack, quantize, and SM86 GEMM tests are present")


def check_benchmarks() -> Result:
    problems = []
    for filename, tokens in BENCHMARK_CONTRACT.items():
        path = ROOT / filename
        text = read_text(path)
        if not text:
            problems.append(f"{filename}: missing")
            continue
        missing = missing_tokens(text, tokens)
        if missing:
            problems.append(f"{filename}: missing " + ", ".join(missing))
    if problems:
        return fail_result("; ".join(problems))
    return pass_result("INT4/INT8 SM86 benchmarks are present and SM86-gated")


CHECKS = (
    Check(
        "setup-sm86-flags",
        "setup.py declares the active compute_86/sm_86 NVCC flag.",
        check_setup_sm86_flags,
    ),
    Check(
        "setup-no-active-sm100-sm120",
        "setup.py has no active SM100/SM120 or compute_100/compute_120 markers.",
        check_no_active_blackwell_flags,
    ),
    Check(
        "setup-int-sources",
        "setup.py lists the new INT4/INT8 CUDA source entries and the files exist.",
        check_int_source_entries,
    ),
    Check(
        "bindings-int-api",
        "bindings.cpp exposes pack_int4, quantize_int8, and INT GEMM APIs.",
        check_bindings,
    ),
    Check(
        "python-wrappers-int-api",
        "qutlass/__init__.py exposes Python wrappers for the active INT APIs.",
        check_python_wrappers,
    ),
    Check(
        "readme-intlass-identity",
        "README.md identifies the active project as Ampere SM86 INT4/INT8 IntLASS.",
        check_readme_identity,
    ),
    Check(
        "tests-int-coverage",
        "INT4/INT8 static test files exist and cover the active API names.",
        check_tests,
    ),
    Check(
        "benchmarks-int-sm86",
        "INT4/INT8 SM86 benchmark files exist and are SM86-gated.",
        check_benchmarks,
    ),
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the static IntLASS Ampere INT4/INT8 migration contract."
    )
    parser.add_argument("--list-checks", action="store_true", help="print checks and exit 0")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit nonzero when any contract check fails",
    )
    return parser.parse_args(argv)


def list_checks() -> None:
    print("IntLASS static contract checks:")
    for check in CHECKS:
        print(f"- {check.name}: {check.description}")


def print_report(results: list[tuple[Check, Result]]) -> None:
    print("IntLASS static contract report")
    print(f"Root: {ROOT}")
    for check, result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"[{status}] {check.name}: {result.detail}")
    passed = sum(1 for _, result in results if result.ok)
    failed = len(results) - passed
    print(f"Summary: {passed} passed, {failed} failed")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.list_checks:
        list_checks()
        return 0

    results = [(check, check.run()) for check in CHECKS]
    print_report(results)
    failed = any(not result.ok for _, result in results)
    if args.strict and failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
