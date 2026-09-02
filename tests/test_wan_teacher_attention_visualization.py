"""CPU tests for the Wan teacher pairwise-attention visualization."""

import importlib.util
from pathlib import Path
import tempfile
import types

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "wanvideo" / "model_inference" / "visualize_wan_teacher_attention.py"
SPEC = importlib.util.spec_from_file_location("wan_teacher_attention_visualization", SCRIPT)
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


def test_frame_token_indices_and_coordinates():
    grid = (3, 2, 4)
    indices = A.frame_token_indices(grid, 1)
    assert indices.tolist() == list(range(8, 16))
    assert A.token_index_to_coordinate(0, grid) == (0, 0, 0)
    assert A.token_index_to_coordinate(8, grid) == (1, 0, 0)
    assert A.token_index_to_coordinate(15, grid) == (1, 1, 3)
    assert A.token_index_to_coordinate(23, grid) == (2, 1, 3)


def test_head_averaged_attention_matches_direct_calculation():
    torch.manual_seed(7)
    q = torch.randn(1, 6, 8)
    k = torch.randn(1, 6, 8)
    query_indices = torch.tensor([1, 4])
    actual = A.head_averaged_attention(q, k, query_indices, num_heads=2)

    qh = q.reshape(1, 6, 2, 4).permute(0, 2, 1, 3)
    kh = k.reshape(1, 6, 2, 4).permute(0, 2, 1, 3)
    selected = qh[:, :, query_indices]
    expected = torch.softmax(selected @ kh.transpose(-1, -2) / 2.0, dim=-1).mean(dim=1)[0]
    assert torch.allclose(actual, expected, atol=1e-6)
    assert torch.allclose(actual.sum(dim=-1), torch.ones(2), atol=1e-6)


def test_log_probability_encoding_is_monotonic_and_bounded():
    probabilities = torch.tensor([0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0])
    encoded = A.encode_log_probabilities(probabilities)
    assert encoded.dtype == torch.uint8
    assert encoded.tolist()[0] == 0
    assert encoded.tolist()[1] == 0
    assert encoded.tolist()[2] == 0
    assert encoded.tolist()[-1] == 255
    assert torch.all(encoded[1:] >= encoded[:-1])

    decoded = A.decode_log_probabilities(encoded[2:])
    expected = probabilities[2:]
    relative_error = (decoded - expected).abs() / expected
    assert torch.all(relative_error < 0.04), relative_error


def test_invalid_indices_are_rejected():
    for frame in (-1, 3):
        try:
            A.frame_token_indices((3, 2, 4), frame)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid frame {frame} was accepted")


def test_runtime_wrapper_preserves_output_and_writes_matrix():
    class FakeAttention:
        def forward(self, q, k, v):
            return q + k + v

    attention = FakeAttention()
    block = types.SimpleNamespace(
        self_attn=types.SimpleNamespace(attn=attention, num_heads=2)
    )
    dit = types.SimpleNamespace(blocks=[block])
    q = torch.randn(1, 8, 8)
    k = torch.randn(1, 8, 8)
    v = torch.randn(1, 8, 8)
    expected = attention.forward(q, k, v)

    with tempfile.TemporaryDirectory() as directory:
        capture = A.WanSelfAttentionCapture(dit, Path(directory), (2, 2, 2), 1)
        capture.install()
        capture.start_step(0, 999.0)
        actual = attention.forward(q, k, v)
        capture.finish_step()
        capture.restore()

        assert torch.equal(actual, expected)
        matrix_path = Path(directory) / capture.step_records[0]["layers"][0]
        with Image.open(matrix_path) as matrix:
            assert matrix.mode == "L"
            assert matrix.size == (8, 4)  # PIL reports (columns, rows)
        assert attention.forward(q, k, v).shape == expected.shape


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    main()
