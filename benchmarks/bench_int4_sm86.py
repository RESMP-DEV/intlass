import argparse
from collections.abc import Callable

import torch
from qutlass import matmul_int4_bf16_tn, pack_int4

BF16_LABEL = "INT4 SM86 BF16 PyTorch baseline"
INT4_LABEL = "INT4 SM86 matmul_int4_bf16_tn"
DEFAULT_SHAPES = (
    (16, 512, 512),
    (64, 512, 512),
    (128, 1024, 1024),
)


def require_sm86(device: int) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the INT4 SM86 benchmark")

    torch.cuda.set_device(device)
    capability = torch.cuda.get_device_capability(device)
    if capability < (8, 6):
        raise SystemExit(
            "INT4 SM86 benchmark requires device capability at least "
            f"(8, 6), got {capability}"
        )
    if capability != (8, 6):
        raise SystemExit(
            "INT4 SM86 benchmark must run on an SM86 CUDA device, got "
            f"sm_{capability[0]}{capability[1]}"
        )


def parse_shape(value: str) -> tuple[int, int, int]:
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be M,N,K or MxNxK")

    try:
        shape = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape entries must be integers") from exc

    if any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError("shape entries must be positive")
    if shape[2] % 2 != 0:
        raise argparse.ArgumentTypeError("K must be even for packed INT4 inputs")
    return shape


def row_scale(x: torch.Tensor) -> torch.Tensor:
    max_abs = x.float().abs().amax(dim=1).clamp(min=1.0e-6)
    return (max_abs / 7.0).contiguous()


def quantize_int4(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    q = torch.round(x.float() / scale[:, None])
    return torch.clamp(q, -8, 7).to(torch.int8).contiguous()


def time_cuda(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


@torch.no_grad()
def run_shape(m: int, n: int, k: int, warmup: int, iters: int) -> tuple[float, float]:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    a = torch.randn((m, k), device=device, dtype=dtype)
    b = torch.randn((n, k), device=device, dtype=dtype)

    a_scale = row_scale(a)
    b_scale = row_scale(b)
    a_packed = pack_int4(quantize_int4(a, a_scale))
    b_packed = pack_int4(quantize_int4(b, b_scale))

    bf16_ms = time_cuda(lambda: torch.nn.functional.linear(a, b), warmup, iters)
    int4_ms = time_cuda(
        lambda: matmul_int4_bf16_tn(a_packed, b_packed, a_scale, b_scale),
        warmup,
        iters,
    )
    return bf16_ms, int4_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="INT4 SM86 GEMM benchmark")
    parser.add_argument(
        "--shape",
        action="append",
        type=parse_shape,
        metavar="M,N,K",
        help="Shape to benchmark; may be repeated. Default: small SM86 shapes.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    if args.iters <= 0:
        raise SystemExit("--iters must be positive")

    require_sm86(args.device)

    shapes = args.shape if args.shape is not None else DEFAULT_SHAPES
    print("INT4 SM86 benchmark")
    print(f"{'M':>8} {'N':>8} {'K':>8} {BF16_LABEL + ' ms':>36} {INT4_LABEL + ' ms':>36}")
    for m, n, k in shapes:
        bf16_ms, int4_ms = run_shape(m, n, k, args.warmup, args.iters)
        print(f"{m:8d} {n:8d} {k:8d} {bf16_ms:36.4f} {int4_ms:36.4f}")


if __name__ == "__main__":
    main()
