"""CPU tests for the warped velocity/residual denoising schedule."""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPT_DIR = os.path.join(ROOT, "examples", "wanvideo", "model_inference")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.abspath(SCRIPT_DIR))

import analyze_velocity_residual_warped as A


def test_t_compute_matches_reference_formula():
    indices = torch.tensor([0.0, 1.0, 7.0, 19.0], dtype=torch.float64)
    step_num = 20
    expected = torch.sqrt(
        1 + A.warp_s - torch.sqrt(
            (1 - (indices / (step_num + 1)) ** (2 * A.warp_rho) - A.warp_s)
            / (1 - 2 * A.warp_s)
        )
    )
    torch.testing.assert_close(A.t_compute(indices, step_num), expected)


def test_warped_schedule_is_descending_and_consistent():
    step_num = 50
    warped_times, sigmas, timesteps = A.make_warped_schedule(step_num)

    assert warped_times.shape == (step_num,)
    assert sigmas.shape == (step_num,)
    assert timesteps.shape == (step_num,)
    assert torch.isfinite(warped_times).all()
    assert torch.isfinite(sigmas).all()
    assert torch.all(warped_times[1:] > warped_times[:-1])
    assert torch.all(sigmas[1:] < sigmas[:-1])
    torch.testing.assert_close(sigmas, 1 - warped_times)
    torch.testing.assert_close(timesteps, 1000 * sigmas)


def test_warped_schedule_uses_step_indices_without_wan_shift():
    step_num = 8
    warped_times, sigmas, _ = A.make_warped_schedule(step_num)
    expected_times = A.t_compute(torch.arange(step_num, dtype=torch.float32), step_num)
    torch.testing.assert_close(warped_times, expected_times)
    torch.testing.assert_close(sigmas, 1 - expected_times)


def test_schedule_rejects_invalid_step_counts():
    for step_num in (0, -1):
        try:
            A.make_warped_schedule(step_num)
        except ValueError:
            pass
        else:
            raise AssertionError(f"step_num={step_num} should be rejected")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    main()
