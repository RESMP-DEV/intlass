import importlib

import pytest


def _to_signed_int4(nibble: int) -> int:
    nibble &= 0x0F
    return nibble - 16 if nibble & 0x08 else nibble


def _reference_pack_int4(torch, x_int4):
    if x_int4.dtype != torch.int8:
        raise TypeError("x_int4 must be torch.int8")
    if x_int4.dim() == 0:
        raise ValueError("x_int4 must have non-empty rank")
    if x_int4.size(-1) % 2 != 0:
        raise ValueError("x_int4 K dimension must be even")
    if torch.any((x_int4 < -8) | (x_int4 > 7)).item():
        raise ValueError("x_int4 values must be in signed INT4 range [-8, 7]")

    output_shape = (*x_int4.shape[:-1], x_int4.size(-1) // 2)
    packed = torch.empty(output_shape, dtype=torch.uint8)
    x_flat = x_int4.reshape(-1, x_int4.size(-1))
    packed_flat = packed.reshape(-1, packed.size(-1))

    for row_idx, row in enumerate(x_flat.tolist()):
        for packed_idx, k in enumerate(range(0, len(row), 2)):
            low_nibble = row[k] & 0x0F
            high_nibble = row[k + 1] & 0x0F
            packed_flat[row_idx, packed_idx] = low_nibble | (high_nibble << 4)

    return packed


def _reference_unpack_int4(torch, packed):
    if packed.dtype != torch.uint8:
        raise TypeError("packed must be torch.uint8")
    if packed.dim() == 0:
        raise ValueError("packed must have non-empty rank")

    output_shape = (*packed.shape[:-1], packed.size(-1) * 2)
    unpacked = torch.empty(output_shape, dtype=torch.int8)
    packed_flat = packed.reshape(-1, packed.size(-1))
    unpacked_flat = unpacked.reshape(-1, unpacked.size(-1))

    for row_idx, row in enumerate(packed_flat.tolist()):
        for packed_idx, byte in enumerate(row):
            unpacked_flat[row_idx, packed_idx * 2] = _to_signed_int4(byte)
            unpacked_flat[row_idx, packed_idx * 2 + 1] = _to_signed_int4(byte >> 4)

    return unpacked


def _require_pack_int4_extension(torch):
    if not torch.cuda.is_available():
        pytest.skip("pack_int4 requires CUDA")
    capability = torch.cuda.get_device_capability()
    if capability != (8, 6):
        pytest.skip("pack_int4 requires an SM86 CUDA device")

    try:
        qutlass_cuda = importlib.import_module("qutlass._CUDA")
    except (ImportError, OSError) as exc:
        pytest.skip(f"qutlass._CUDA is unavailable: {exc}")

    pack_int4 = getattr(qutlass_cuda, "pack_int4", None)
    if pack_int4 is None:
        pytest.skip("qutlass._CUDA.pack_int4 is unavailable")
    return pack_int4


def test_reference_pack_int4_uses_even_low_odd_high_nibbles() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    x_int4 = torch.tensor(
        [
            [-8, -1, 0, 7],
            [7, 0, -1, -8],
        ],
        dtype=torch.int8,
    )
    expected = torch.tensor(
        [
            [0xF8, 0x70],
            [0x07, 0x8F],
        ],
        dtype=torch.uint8,
    )

    actual = _reference_pack_int4(torch, x_int4)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_reference_unpack_int4_restores_signed_values() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    expected = torch.tensor(
        [
            [-8, -1, 0, 7],
            [7, 0, -1, -8],
        ],
        dtype=torch.int8,
    )
    packed = _reference_pack_int4(torch, expected)

    actual = _reference_unpack_int4(torch, packed)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_cuda_pack_int4_matches_cpu_reference() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    pack_int4 = _require_pack_int4_extension(torch)
    x_int4 = torch.tensor(
        [
            [-8, -1, 0, 7],
            [7, 0, -1, -8],
        ],
        dtype=torch.int8,
    )
    expected = _reference_pack_int4(torch, x_int4)

    actual = pack_int4(x_int4.cuda()).cpu()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
