"""CPU tests for the agent harness loop: ledger, manifest, verify, review."""

import json

import pytest

import cache_head_harness as h
from cache_head_model import CacheHead, CacheHeadConfig, save_cache_head


# ═══════════════════════════════════════════════════════════════
# Ledger
# ═══════════════════════════════════════════════════════════════

def test_ledger_append_and_verify_chain(tmp_path):
    ledger = h.Ledger(tmp_path / "ledger.jsonl")
    e1 = ledger.append({"kind": "a", "n": 1})
    e2 = ledger.append({"kind": "b", "n": 2})
    e3 = ledger.append({"kind": "c", "n": 3})
    assert e1["prev_entry_id"] == h.Ledger.GENESIS
    assert e2["prev_entry_id"] == e1["entry_id"]
    assert e3["prev_entry_id"] == e2["entry_id"]
    ok, problems = ledger.verify()
    assert ok, problems
    assert len(ledger.read()) == 3


def test_ledger_tamper_detection(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = h.Ledger(path)
    ledger.append({"kind": "a"})
    ledger.append({"kind": "b"})
    # Tamper with an entry's content.
    lines = path.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[0])
    entry["kind"] = "TAMPERED"
    lines[0] = json.dumps(entry)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok, problems = ledger.verify()
    assert not ok
    assert any("tampered" in p for p in problems)


# ═══════════════════════════════════════════════════════════════
# Manifest (controller)
# ═══════════════════════════════════════════════════════════════

def test_manifest_emit_and_lock(tmp_path):
    m1 = h.emit_manifest("dmd", "prompts.jsonl", tmp_path, seeds=[0, 1, 2])
    # Re-emitting the identical manifest is idempotent.
    m2 = h.emit_manifest("dmd", "prompts.jsonl", tmp_path, seeds=[0, 1, 2])
    assert m1["manifest_id"] == m2["manifest_id"]
    assert m1["locked"] is True
    assert m1["arm"] == "dmd"
    assert m1["schedule"]["full_step_indices"] == [1, 2, 6, 10, 14]
    # A different locked manifest is refused.
    with pytest.raises(RuntimeError):
        h.emit_manifest("dmd", "prompts.jsonl", tmp_path, seeds=[0, 1, 2], cfg_scale=7.0)


def test_manifest_unknown_hypothesis_rejected(tmp_path):
    with pytest.raises(ValueError):
        h.emit_manifest("not_a_hypothesis", "p.jsonl", tmp_path)


def test_manifest_carry_previous_has_no_arm(tmp_path):
    m = h.emit_manifest("carry_previous", "p.jsonl", tmp_path)
    assert m["arm"] is None


# ═══════════════════════════════════════════════════════════════
# Verification invariants
# ═══════════════════════════════════════════════════════════════

def test_verify_invariants_cpu_checks(monkeypatch, tmp_path):
    monkeypatch.setattr(h, "run_pytest", lambda: True)
    # A checkpoint containing only CacheHead weights.
    head = CacheHead(CacheHeadConfig())
    head.eval()
    ckpt = tmp_path / "head.ckpt"
    save_cache_head(head, CacheHeadConfig(), ckpt)

    report = h.verify_invariants(checkpoint=ckpt)
    assert report["zero_init_equals_carry"] is True
    assert report["schedule_5_full_10_head"] is True
    assert report["no_fake_score_in_export"] is True
    assert report["unit_tests_passed"] is True
    assert report["frozen_wan_no_trainable"].startswith("not_checked")
    assert report["optimizer_only_cachehead"].startswith("not_checked")


def test_verify_optimizer_only_cachehead(monkeypatch, tmp_path):
    monkeypatch.setattr(h, "run_pytest", lambda: True)
    head = CacheHead(CacheHeadConfig())
    import torch

    class FakeTrainer:
        def __init__(self, head):
            self.head = head
            self.head_opt = torch.optim.AdamW(head.parameters(), lr=1e-3)

    report = h.verify_invariants(trainer=FakeTrainer(head), checkpoint=tmp_path / "x.ckpt")
    # checkpoint missing -> no_fake_score check reports not_checked
    assert report["optimizer_only_cachehead"] is True


# ═══════════════════════════════════════════════════════════════
# Review decision
# ═══════════════════════════════════════════════════════════════

def _good_metrics():
    return {
        "finite_checks": [1.0, 1.0, 1.0],
        "schedule_violation": False,
        "late_step_explosion": False,
        "velocity_error": 0.05,
        "clip_alignment": 0.30,
        "temporal_flicker": 0.10,
        "speedup": 2.0,
        "pca_complete": True,
    }


def test_decide_promotes_when_all_thresholds_met():
    baselines = {"carry_previous": {"velocity_error": 0.10, "clip_alignment": 0.28, "temporal_flicker": 0.12}}
    decision, reasons = h.decide(_good_metrics(), baselines)
    assert decision == "promote"
    assert reasons == []


def test_decide_rejects_low_velocity_improvement():
    metrics = _good_metrics()
    metrics["velocity_error"] = 0.095  # only 5% improvement over 0.10
    baselines = {"carry_previous": {"velocity_error": 0.10, "clip_alignment": 0.28, "temporal_flicker": 0.12}}
    decision, reasons = h.decide(metrics, baselines)
    assert decision == "not_promoted"
    assert any("velocity error" in r for r in reasons)


def test_decide_rejects_regression_and_slowdown():
    metrics = _good_metrics()
    metrics["clip_alignment"] = 0.20   # regression vs 0.28
    metrics["speedup"] = 1.2           # below 1.5x
    metrics["late_step_explosion"] = True
    metrics["pca_complete"] = False
    baselines = {"carry_previous": {"velocity_error": 0.10, "clip_alignment": 0.28, "temporal_flicker": 0.12}}
    decision, reasons = h.decide(metrics, baselines)
    assert decision == "not_promoted"
    assert len(reasons) >= 4


# ═══════════════════════════════════════════════════════════════
# Next arm selection
# ═══════════════════════════════════════════════════════════════

def test_select_next_arm_skips_run_hypotheses(tmp_path):
    ledger = h.Ledger(tmp_path / "ledger.jsonl")
    ledger.append({"manifest": {"hypothesis": {"id": "carry_previous"}}, "reviewer_decision": {"decision": "done"}})
    ledger.append({"manifest": {"hypothesis": {"id": "residual_regression_huber"}}, "reviewer_decision": {"decision": "done"}})
    nxt = h.select_next_arm(ledger)
    assert isinstance(nxt, dict)
    assert nxt["id"] == "dmd"


def test_select_next_arm_promotes_when_all_ran(tmp_path):
    ledger = h.Ledger(tmp_path / "ledger.jsonl")
    for hypothesis in h.HYPOTHESES:
        ledger.append({"manifest": {"hypothesis": hypothesis}, "reviewer_decision": {"decision": "done"}})
    assert h.select_next_arm(ledger) == "promote_to_larger_run"
