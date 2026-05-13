import importlib

import pytest


def _scale_view(scale, x):
    if scale.numel() == 1:
        return scale.reshape((1,) * x.dim())
    return scale.reshape(*x.shape[:-1], 1)


def _reference_quantize_int8(torch, x, scale):
    x_f32 = x.to(torch.float32)
    scale_f32 = _scale_view(scale.to(torch.float32), x_f32)
    return torch.clamp(torch.round(x_f32 / scale_f32), -128, 127).to(torch.int8)


def _require_quantize_int8_extension(torch):
    if not torch.cuda.is_available():
        pytest.skip("quantize_int8 requires CUDA")
    capability = torch.cuda.get_device_capability()
    if capability < (8, 6):
        pytest.skip("quantize_int8 requires CUDA device capability >= 8.6")

    try:
        qutlass_cuda = importlib.import_module("qutlass._CUDA")
    except (ImportError, OSError) as exc:
        pytest.skip(f"qutlass._CUDA is unavailable: {exc}")

    quantize_int8 = getattr(qutlass_cuda, "quantize_int8", None)
    if quantize_int8 is None:
        pytest.skip("qutlass._CUDA.quantize_int8 is unavailable")
    return quantize_int8


def test_reference_quantize_int8_per_tensor_scale() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    x = torch.tensor(
        [
            [
                -258.0,
                -257.0,
                -255.0,
                -2.0,
                -1.0,
                0.0,
                1.0,
                2.0,
                255.0,
                256.0,
                257.0,
            ]
        ],
        dtype=torch.float32,
    )
    scale = torch.tensor([2.0], dtype=torch.float32)
    expected = torch.tensor(
        [[-128, -128, -128, -1, 0, 0, 0, 1, 127, 127, 127]],
        dtype=torch.int8,
    )

    actual = _reference_quantize_int8(torch, x, scale)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_reference_quantize_int8_per_row_scale() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    x = torch.tensor(
        [
            [-12.0, -9.0, -4.0, -1.0, 0.0, 1.0, 4.0, 9.0],
            [-300.0, -192.0, -3.0, 3.0, 63.0, 64.0, 65.0, 384.0],
            [-1.5, -0.75, -0.25, 0.25, 0.75, 1.5, 2.25, 3.75],
        ],
        dtype=torch.float32,
    )
    scale = torch.tensor([3.0, 0.5, 0.25], dtype=torch.float32)
    expected = torch.tensor(
        [
            [-4, -3, -1, 0, 0, 0, 1, 3],
            [-128, -128, -6, 6, 126, 127, 127, 127],
            [-6, -3, -1, 1, 3, 6, 9, 15],
        ],
        dtype=torch.int8,
    )

    actual = _reference_quantize_int8(torch, x, scale)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("x_values", "scale_values", "x_dtype", "scale_dtype"),
    [
        (
            [[-258.0, -255.0, -1.0, 0.0, 1.0, 2.0, 255.0, 257.0]],
            [2.0],
            "float32",
            "float32",
        ),
        (
            [
                [-12.0, -9.0, -4.0, 0.0, 4.0, 9.0],
                [-300.0, -3.0, 3.0, 63.0, 64.0, 384.0],
                [-1.5, -0.75, 0.75, 1.5, 2.25, 3.75],
            ],
            [3.0, 0.5, 0.25],
            "bfloat16",
            "bfloat16",
        ),
    ],
    ids=["per_tensor_scale", "per_row_scale"],
)
def test_cuda_quantize_int8_matches_cpu_reference(
    x_values: list[list[float]],
    scale_values: list[float],
    x_dtype: str,
    scale_dtype: str,
) -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    quantize_int8 = _require_quantize_int8_extension(torch)
    x = torch.tensor(x_values, dtype=getattr(torch, x_dtype))
    scale = torch.tensor(scale_values, dtype=getattr(torch, scale_dtype))
    expected = _reference_quantize_int8(torch, x, scale)

    actual = quantize_int8(x.cuda(), scale.cuda()).cpu()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
