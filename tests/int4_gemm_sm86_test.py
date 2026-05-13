import pytest


def _require_sm86_or_newer(torch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("matmul_int4_bf16_tn requires CUDA")
    capability = torch.cuda.get_device_capability()
    if capability < (8, 6):
        pytest.skip("matmul_int4_bf16_tn requires CUDA device capability >= 8.6")


def _rand_int4(torch, shape: tuple[int, int], seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randint(-8, 8, shape, dtype=torch.int8, generator=generator)


def _pack_int4_python(torch, x_int4):
    if x_int4.dtype != torch.int8:
        raise TypeError("x_int4 must be torch.int8")
    if x_int4.dim() != 2:
        raise ValueError("x_int4 must be 2D")
    if x_int4.size(1) % 2 != 0:
        raise ValueError("x_int4 K dimension must be even")
    if torch.any((x_int4 < -8) | (x_int4 > 7)).item():
        raise ValueError("x_int4 values must be in signed INT4 range [-8, 7]")

    packed = torch.empty((x_int4.size(0), x_int4.size(1) // 2), dtype=torch.uint8)
    for row_idx, row in enumerate(x_int4.tolist()):
        for k in range(0, len(row), 2):
            low_nibble = row[k] & 0x0F
            high_nibble = row[k + 1] & 0x0F
            packed[row_idx, k // 2] = low_nibble | (high_nibble << 4)
    return packed


def _scale_view(scale, rows: int):
    if scale.numel() == 1:
        return scale.reshape(1, 1)
    return scale.reshape(rows, 1)


def _dequantize_rows(torch, x_int4, scale):
    return x_int4.to(torch.float32) * _scale_view(scale.to(torch.float32), x_int4.size(0))


def _reference_int4_bf16_tn(
    torch,
    a_int4,
    b_int4,
    a_scale,
    b_scale,
):
    a_dequant = _dequantize_rows(torch, a_int4, a_scale)
    b_dequant = _dequantize_rows(torch, b_int4, b_scale)
    return (a_dequant @ b_dequant.T).to(torch.bfloat16)


@pytest.mark.parametrize(
    ("m", "n", "k", "a_scale_values", "a_scale_dtype", "b_scale_values", "b_scale_dtype"),
    [
        (
            3,
            4,
            18,
            [0.125],
            "float32",
            [0.25],
            "float32",
        ),
        (
            5,
            3,
            34,
            [0.125, 0.25, 0.5, 1.0, 0.0625],
            "bfloat16",
            [0.5, 0.25, 0.125],
            "bfloat16",
        ),
    ],
    ids=["per_tensor_scales", "per_row_bf16_scales"],
)
def test_matmul_int4_bf16_tn_matches_dequantized_reference(
    m: int,
    n: int,
    k: int,
    a_scale_values: list[float],
    a_scale_dtype: str,
    b_scale_values: list[float],
    b_scale_dtype: str,
) -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    _require_sm86_or_newer(torch)

    from qutlass import matmul_int4_bf16_tn

    device = torch.device("cuda")
    a_int4 = _rand_int4(torch, (m, k), seed=2026)
    b_int4 = _rand_int4(torch, (n, k), seed=2057)
    a_packed = _pack_int4_python(torch, a_int4)
    b_packed = _pack_int4_python(torch, b_int4)
    a_scale = torch.tensor(a_scale_values, dtype=getattr(torch, a_scale_dtype))
    b_scale = torch.tensor(b_scale_values, dtype=getattr(torch, b_scale_dtype))

    expected = _reference_int4_bf16_tn(torch, a_int4, b_int4, a_scale, b_scale).to(device)
    actual = matmul_int4_bf16_tn(
        a_packed.to(device),
        b_packed.to(device),
        a_scale.to(device),
        b_scale.to(device),
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
