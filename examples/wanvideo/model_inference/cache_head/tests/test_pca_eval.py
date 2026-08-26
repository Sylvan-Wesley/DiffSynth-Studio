"""CPU tests for the shared-PCA trajectory-difference analysis (no Wan needed)."""

import json
import os

import numpy as np
import pytest

from pca_trajectory_eval import (
    build_shared_pca,
    make_trajectory_plot,
    save_artifacts,
    trajectory_metrics,
)

METHODS = ["full_wan", "carry_previous", "arm"]
P, STEPS = 2, 16  # 16 states per 15-step rollout


def _synthetic_runs(seed=0, identical_arms=False):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((P, STEPS, 16, 2, 3, 4)).astype(np.float32)
    runs = {METHODS[0]: base}
    if identical_arms:
        runs[METHODS[1]] = base.copy()
        runs[METHODS[2]] = base.copy()
    else:
        runs[METHODS[1]] = base + 0.5 * rng.standard_normal(base.shape).astype(np.float32)
        runs[METHODS[2]] = base + 0.1 * rng.standard_normal(base.shape).astype(np.float32)
    return runs


def test_build_shared_pca_shapes_and_single_basis():
    runs = _synthetic_runs()
    fit, coords = build_shared_pca(runs, pca_seed=0)
    # One shared fit over the concatenated runs.
    all_traj = np.concatenate([runs[m] for m in METHODS], axis=0)
    assert fit["coordinates"].shape == (3 * P, STEPS, 2)
    assert coords["full_wan"].shape == (P, STEPS, 2)
    assert coords["carry_previous"].shape == (P, STEPS, 2)
    assert coords["arm"].shape == (P, STEPS, 2)
    assert fit["components"].shape == (np.prod(runs["full_wan"].shape[2:]), 2)
    assert fit["explained_variance_ratio"].shape == (2,)
    assert np.all(np.isfinite(fit["coordinates"]))


def test_shared_basis_is_joint():
    """Coords must be the split of a single fit over ALL methods (shared basis),
    not per-method fits."""
    runs = _synthetic_runs()
    fit, coords = build_shared_pca(runs, pca_seed=3)
    all_traj = np.concatenate([runs[m] for m in METHODS], axis=0)
    expect = fit_shared_pca_2d_reference(all_traj, random_seed=3)
    assert np.allclose(coords["full_wan"], expect["coordinates"][:P])
    assert np.allclose(coords["arm"], expect["coordinates"][2 * P:])


def fit_shared_pca_2d_reference(trajectories, random_seed=0):
    import torch

    from analyze_geometric_trajectory import _validate_trajectories, _pca_compute_device, _fork_rng_devices

    trajectories = _validate_trajectories(trajectories)
    runs, steps = trajectories.shape[:2]
    samples = trajectories.reshape(runs * steps, -1)
    pca_device = _pca_compute_device()
    x = torch.from_numpy(samples).to(pca_device)
    mean = x.mean(dim=0)
    centered = x - mean
    q = min(8, centered.shape[0], centered.shape[1])
    with torch.random.fork_rng(devices=_fork_rng_devices(pca_device)):
        torch.manual_seed(random_seed)
        _, singular_values, right_vectors = torch.pca_lowrank(centered, q=q, center=False, niter=2)
    components = right_vectors[:, :2]
    coordinates = centered @ components
    for component_id in range(2):
        pivot = torch.argmax(components[:, component_id].abs())
        if components[pivot, component_id] < 0:
            components[:, component_id].neg_()
            coordinates[:, component_id].neg_()
    denominator = max(centered.shape[0] - 1, 1)
    explained_variance = singular_values[:2].square() / denominator
    total_variance = centered.square().sum() / denominator
    return {
        "coordinates": coordinates.reshape(runs, steps, 2).cpu().numpy(),
        "explained_variance_ratio": (explained_variance / total_variance).cpu().numpy(),
    }


def test_shared_pca_deterministic_seed_sign():
    runs = _synthetic_runs(seed=1)
    f1, c1 = build_shared_pca(runs, pca_seed=7)
    f2, c2 = build_shared_pca(runs, pca_seed=7)
    assert np.allclose(c1["arm"], c2["arm"])
    assert np.array_equal(f1["components"], f2["components"])


def test_trajectory_metrics_identical_to_full_are_zero():
    runs = _synthetic_runs(identical_arms=True)
    _, coords = build_shared_pca(runs, pca_seed=0)
    metrics = trajectory_metrics(runs, coords)
    for m in (METHODS[1], METHODS[2]):
        md = metrics[m]
        assert md["pca_pointwise_distance"]["mean"] == pytest.approx(0.0, abs=1e-5)
        assert md["pca_terminal_distance"]["mean"] == pytest.approx(0.0, abs=1e-5)
        assert md["latent_l2"]["mean"] == pytest.approx(0.0, abs=1e-5)
        assert md["latent_cosine"]["final_step_mean"] == pytest.approx(1.0, abs=1e-5)
        assert md["latent_relative_drift"]["mean"] == pytest.approx(0.0, abs=1e-5)


def test_trajectory_metrics_nonidentical_positive():
    runs = _synthetic_runs(identical_arms=False)
    _, coords = build_shared_pca(runs, pca_seed=0)
    metrics = trajectory_metrics(runs, coords)
    for m in (METHODS[1], METHODS[2]):
        md = metrics[m]
        for k in ("pca_pointwise_distance", "pca_cumulative_path_distance",
                  "pca_terminal_distance", "latent_l2", "latent_relative_drift"):
            assert np.isfinite(md[k]["mean"]) and md[k]["mean"] > 0
        assert md["latent_cosine"]["mean"] < 1.0


def test_save_artifacts_round_trip(tmp_path):
    runs = _synthetic_runs()
    fit, coords = build_shared_pca(runs, pca_seed=0)
    metrics = trajectory_metrics(runs, coords)
    timesteps = np.linspace(999, 1, STEPS - 1)
    sigmas = timesteps / 1000.0
    npz = str(tmp_path / "pca_trajectory.npz")
    png = str(tmp_path / "pca_trajectory.png")
    metrics_path = str(tmp_path / "trajectory_metrics.json")
    make_trajectory_plot(coords, fit["explained_variance_ratio"], [0, 1], [2, 3], png)
    assert os.path.isfile(png)
    save_artifacts(
        npz, png, metrics_path,
        fit=fit, per_method_coords=coords, method_labels=METHODS,
        prompt_ids=["0", "1"], seeds=[0], timesteps=timesteps, sigmas=sigmas,
        metrics=metrics,
    )
    with np.load(npz, allow_pickle=False) as d:
        assert d["coordinates"].shape == (len(METHODS), P, 1, STEPS, 2)
        assert np.array_equal(d["method_labels"], np.asarray(METHODS))
        assert np.array_equal(d["seeds"], np.asarray([0]))
        assert d["step_indices"].shape == (STEPS,)
        assert np.all(np.isfinite(d["coordinates"]))
    with open(metrics_path) as fh:
        assert isinstance(json.load(fh), dict)
