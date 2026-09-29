import pytest

from llmba.measure import (
    FAIL,
    INSUFFICIENT,
    PASS,
    bootstrap_mdape,
    grade,
    grade_negative_control,
    grid,
    grid_check,
    knee,
    knee_agrees,
    predict,
)
from llmba.specs import HardwareSpec, load_twin


def rows(err=0.05, invalid=0, knee_shift=1.0):
    out = []
    for c in (128, 1024):
        for b in (1, 2, 4, 8, 16, 32, 64):
            meas = 0.01 * max(1.0, b / 8) * (1 + c / 4096)
            pred = 0.01 * max(1.0, b / (8 * knee_shift)) * (1 + c / 4096) * (1 + err)
            out.append({"batch": b, "context": c, "measured_s": meas, "predicted_s": pred,
                        "control_s": meas / 4, "valid": True})
    for r in out[:invalid]:
        r["valid"] = False
    return out


def test_knee_rule():
    assert knee({1: 1.0, 2: 1.2, 4: 1.6}) == 4
    assert knee({1: 1.0, 2: 1.1}) is None
    assert knee_agrees(8, 16) and knee_agrees(None, None)
    assert not knee_agrees(8, 32) and not knee_agrees(8, None)


def test_grade_pass_fail_insufficient():
    assert grade(rows(0.05), 14)["verdict"] == PASS
    assert grade(rows(0.6), 14)["verdict"] == FAIL  # MdAPE too large
    g = grade(rows(0.0, knee_shift=4.0), 14)  # right magnitude, knee two steps off
    assert g["verdict"] == FAIL and not g["knee_agrees"]
    assert grade(rows(0.05, invalid=4), 14)["verdict"] == INSUFFICIENT  # 10/14 < 80%


def test_negative_control_rejected_only_when_wrong():
    assert grade_negative_control(rows())["rejected"]  # 75% error everywhere
    good = [{**r, "control_s": r["measured_s"] * 1.1} for r in rows()]
    assert not grade_negative_control(good)["rejected"]


def test_bootstrap_deterministic_and_bracketing():
    ape = [0.1, 0.2, 0.3, 0.15, 0.05, 0.4]
    a, b = bootstrap_mdape(ape), bootstrap_mdape(ape)
    assert a == b and a[1] <= a[0] <= a[2]


def test_frozen_final_grid_meets_rule():
    tw = load_twin("tiny-final")
    hw = HardwareSpec(name="local-mps", peak_tflops={"fp16": 1.08}, hbm_gb=5.7, hbm_gbps=88.0,
                      compute_eff=1.0, mem_eff=1.0, kernel_overhead_us=6.0)
    assert len(grid(tw)) == 45 and grid_check(tw, hw)["ok"]
    t1, share = predict(tw, hw, 1, 128)
    assert share > 0.9 and t1 == pytest.approx(0.0071, rel=0.1)


def test_twin_runs_on_cpu_and_matches_shapes():
    torch = pytest.importorskip("torch")
    from llmba.measure import TwinModel
    from llmba.specs import ModelSpec, Twin
    m = ModelSpec(name="t", n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, head_dim=16,
                  d_ff=96, vocab=32)
    tw = Twin(model=m, batches=(1, 2), contexts=(8,), dtype="fp32")
    tm = TwinModel(tw, "cpu")
    caches = tm.caches(2, 8)
    out = tm.step(torch.tensor([1, 2]), caches, 7)
    assert out.shape == (2, 32) and torch.isfinite(out).all()


def test_knee_unresolvable_when_baseline_invalid():
    rs = rows(0.0, knee_shift=100.0)  # predicted knee never reached, measured knee at 16
    assert not grade(rs, 14)["knee_agrees"]
    rs[0]["valid"] = False  # batch 1 at the shortest context fails the contention rule
    g = grade(rs, 14)
    assert g["knee_context"] == 128 and not g["knee_resolvable"] and g["verdict"] == FAIL


def test_negative_control_empty():
    assert grade_negative_control([{**r, "valid": False} for r in rows()])["rejected"] is False
