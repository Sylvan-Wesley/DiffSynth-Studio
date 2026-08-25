"""CPU tests for shared PCA projection of latent trajectories."""

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPT_DIR = os.path.join(ROOT, "examples", "wanvideo", "model_inference")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.abspath(SCRIPT_DIR))

import analyze_geometric_trajectory as A


def test_fork_rng_uses_an_explicit_cuda_index():
    assert A._fork_rng_devices(torch.device("cpu")) == []
    assert A._fork_rng_devices(torch.device("cuda:2")) == [2]
    with patch.object(A.torch.cuda, "current_device", return_value=3):
        assert A._fork_rng_devices(torch.device("cuda")) == [3]


def test_run_cache_reuses_only_matching_complete_runs():
    trajectory = np.arange(24, dtype=np.float32).reshape(3, 2, 2, 2)
    final_latents = trajectory[-1]
    with TemporaryDirectory() as temporary_directory:
        cache_path = Path(temporary_directory) / "seed_0007.npz"
        A._save_run_cache(
            cache_path,
            seed=7,
            fingerprint="configuration-a",
            trajectory=trajectory,
            final_latents=final_latents,
        )
        loaded = A._load_cached_run(
            cache_path,
            seed=7,
            fingerprint="configuration-a",
            expected_steps=3,
            expected_latent_shape=(2, 2, 2),
        )
        assert loaded is not None
        assert np.array_equal(loaded[0], trajectory)
        assert np.array_equal(loaded[1], final_latents)
        assert A._load_cached_run(
            cache_path,
            seed=7,
            fingerprint="configuration-b",
            expected_steps=3,
            expected_latent_shape=(2, 2, 2),
        ) is None


def test_shared_pca_shapes_and_centering():
    rng = np.random.default_rng(0)
    trajectories = rng.standard_normal((3, 4, 2, 3)).astype(np.float32)
    result = A.fit_shared_pca_2d(trajectories, device="cpu")

    assert result["coordinates"].shape == (3, 4, 2)
    assert result["components"].shape == (6, 2)
    assert result["mean"].shape == (6,)
    assert result["explained_variance_ratio"].shape == (2,)
    assert np.allclose(result["coordinates"].reshape(-1, 2).mean(axis=0), 0.0, atol=1e-5)
    assert np.all(result["explained_variance_ratio"] >= 0)
    assert result["explained_variance_ratio"].sum() <= 1.0 + 1e-5


def test_shared_basis_retains_time_order_and_is_deterministic():
    # All samples advance in the first feature, so PC1 must preserve the
    # increasing time order after the helper's deterministic sign convention.
    trajectories = np.zeros((2, 5, 2), dtype=np.float32)
    trajectories[0, :, 0] = np.arange(5, dtype=np.float32)
    trajectories[1, :, 0] = np.arange(5, dtype=np.float32) + 0.5

    first = A.fit_shared_pca_2d(trajectories, random_seed=7, device="cpu")
    second = A.fit_shared_pca_2d(trajectories, random_seed=7, device="cpu")

    assert np.all(np.diff(first["coordinates"][0, :, 0]) > 0)
    assert np.allclose(first["coordinates"], second["coordinates"], atol=1e-6)
    assert np.allclose(first["components"], second["components"], atol=1e-6)


def test_pca_rejects_invalid_trajectory_inputs():
    for invalid in (np.zeros((1, 2), dtype=np.float32), np.full((1, 2, 2), np.nan, dtype=np.float32)):
        try:
            A.fit_shared_pca_2d(invalid, device="cpu")
        except ValueError:
            pass
        else:
            raise AssertionError("invalid trajectory input was accepted")


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    main()
