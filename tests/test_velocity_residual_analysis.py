"""CPU smoke tests for the velocity/residual analysis primitives.

Exercises the pure functions in
``examples/wanvideo/model_inference/analyze_velocity_residual.py`` on synthetic
tensors — no model load, no GPU. Mirrors the plain-assert ``main()`` style of the
other ``tests/*.py`` files.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPT_DIR = os.path.join(ROOT, "examples", "wanvideo", "model_inference")
sys.path.insert(0, ROOT)          # make the `diffsynth` package importable
sys.path.insert(0, os.path.abspath(SCRIPT_DIR))

import analyze_velocity_residual as A


_RNG = np.random.default_rng(0)


def _rand(*shape):
    return _RNG.standard_normal(shape).astype(np.float32)


def test_cosine_pearson():
    v = _rand(4, 3, 5, 5)
    assert abs(A.cosine(v, v) - 1.0) < 1e-5, "cosine of identical vectors != 1"
    assert abs(A.pearson(v, v) - 1.0) < 1e-5, "pearson of identical vectors != 1"
    assert abs(A.pearson(v, -v) + 1.0) < 1e-5, "pearson of negated vector != -1"

    # two vectors orthogonal in the uncentered sense -> cosine ~ 0
    a = _rand(4, 3, 5, 5)
    b = _rand(4, 3, 5, 5)
    af, bf = A._flatten(a), A._flatten(b)
    orth = af - (np.dot(af, bf) / np.dot(bf, bf)) * bf
    assert abs(np.dot(orth, bf)) < 1e-3 * np.linalg.norm(orth) * np.linalg.norm(bf), \
        "constructed vectors not orthogonal"
    assert abs(A.cosine(orth.reshape(a.shape), b)) < 1e-4, \
        "cosine of orthogonal vectors != 0"


def test_relative_norm():
    v = _rand(4, 3, 5, 5)
    assert abs(A.relative_norm(2.0 * v, v) - 2.0) < 1e-4, "relative_norm(2v, v) != 2"


def test_per_frame_rank_rank1():
    C, F, H, W = 4, 2, 6, 6
    v = np.zeros((C, F, H, W), dtype=np.float32)
    for f in range(F):
        col = (np.arange(1, C + 1, dtype=np.float32) * (f + 1))
        v[:, f, :, :] = np.broadcast_to(col[:, None, None], (C, H, W))
    ranks, stable, svals = A.per_frame_rank(v)
    assert ranks.shape == (F,), ranks.shape
    assert stable.shape == (F,), stable.shape
    assert svals.shape == (F, C), svals.shape
    assert np.all(ranks == 1), f"rank-1 frames gave ranks {ranks}"
    assert np.allclose(stable, 1.0, atol=1e-2), f"stable rank {stable}"


def test_per_frame_rank_full():
    C, F, H, W = 4, 2, 6, 6
    v = _rand(C, F, H, W)
    ranks, stable, svals = A.per_frame_rank(v)
    assert np.all(ranks == C), f"random frames gave ranks {ranks}"


def test_projection_1d():
    v = _rand(4, 3, 5, 5)
    # residual along v_pre -> fraction ~1
    assert abs(A.project_fraction_1d(2.5 * v, v) - 1.0) < 1e-5
    # residual orthogonal to v_pre -> fraction ~0
    vf = A._flatten(v)
    w = _rand(4, 3, 5, 5)
    wf = A._flatten(w)
    orth = wf - float(np.dot(vf, wf)) / float(np.dot(vf, vf)) * vf
    assert abs(A.project_fraction_1d(orth.reshape(v.shape), v)) < 1e-4


def test_growing_span():
    C, F, H, W = 4, 2, 6, 6
    v0 = _rand(C, F, H, W)
    v1 = _rand(C, F, H, W)
    span = A.GrowingSpan()
    span.add(v0)
    # residual along v0 (already in span) -> ~1
    assert abs(span.project_fraction(3.0 * v0) - 1.0) < 1e-5
    span.add(v1)
    # residual in span{v0, v1} -> ~1
    r = 2.0 * v0 - 1.5 * v1
    assert abs(span.project_fraction(r) - 1.0) < 1e-4
    # residual orthogonal to both -> ~0
    v2 = _rand(C, F, H, W)
    assert abs(span.project_fraction(v2) - 1.0) > 1e-2  # sanity: v2 not in span
    assert span.project_fraction(v2) < 1.0


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    main()
