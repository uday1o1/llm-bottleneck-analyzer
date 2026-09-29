import dataclasses
import json

import pytest

from llmba.analyze import request
from llmba.intercepts import (
    FOUND,
    INCONCLUSIVE,
    _first_true,
    decode_knee,
    max_throughput,
    recommend,
    tier2_breakeven,
    tp_saturation,
    what_ifs,
)
from llmba.specs import Workload, load_hardware, load_model
from llmba.sweep import run_grid, sweep, verify_manifest

L8 = load_model("llama-3.1-8b")
H100 = load_hardware("h100-sxm")


def test_first_true_bisection():
    assert _first_true(lambda x: x >= 37, 1, 1000) == 37
    assert _first_true(lambda x: x >= 1, 1, 10) == 1
    assert _first_true(lambda x: False, 1, 10) is None


def test_gemm_knee_near_analytic_ridge_for_dense_short_context():
    k = decode_knee(L8, H100, Workload(prompt_len=128))
    assert k["status"] == FOUND
    assert k["analytic_gemm_ridge"] / 2 <= k["gemm_knee"] <= 2 * k["analytic_gemm_ridge"]


def test_knee_inconclusive_when_attention_dominates():
    """GQA decode attention has fixed intensity, so long context never turns compute-bound."""
    k = decode_knee(L8, H100, Workload(prompt_len=4096))
    assert k["status"] == INCONCLUSIVE and k["value"] is None and k["range"][1] >= 1


def test_step_knee_monotone_in_batch():
    wl = Workload(prompt_len=128)
    k = decode_knee(L8, H100, wl)["value"]
    from llmba.intercepts import _decode_share
    shares = [_decode_share(L8, H100, dataclasses.replace(wl, batch=b), 128,
                            lambda _, e: e["compute"]) for b in (1, k - 1, k, 2 * k)]
    assert shares == sorted(shares) and shares[1] < 0.5 <= shares[2]


def test_moe_pushes_gemm_knee_past_dense():
    wl = Workload(prompt_len=128, tp=2)
    dense = decode_knee(L8, H100, wl)["gemm_knee"]
    moe = decode_knee(load_model("mixtral-8x7b"), H100, wl)["gemm_knee"]
    assert moe is None or moe > 2 * dense


def test_tp_saturation_skips_nonfitting():
    ds = load_model("deepseek-v3")
    r = tp_saturation(ds, load_hardware("h200"), Workload(tp=8, weight_dtype="fp8"))
    assert set(r["tpot_s"]) == {8}


def test_tier2_never_lowers_throughput_and_breakeven():
    wl = Workload(batch=4, prompt_len=32768, output_len=256)
    base = max_throughput(L8, H100, wl)
    for gbps in (8.0, 900.0):
        t = dataclasses.replace(H100, tier2_gb=480.0, tier2_gbps=gbps)
        assert max_throughput(L8, t, wl) >= base * (1 - 1e-9)
    be = tier2_breakeven(L8, H100, wl)
    assert be["status"] == FOUND
    fast = dataclasses.replace(H100, tier2_gb=480.0, tier2_gbps=be["value"] * 1.2)
    slow = dataclasses.replace(H100, tier2_gb=480.0, tier2_gbps=be["value"] * 0.5)
    assert max_throughput(L8, fast, wl) > base >= max_throughput(L8, slow, wl) * (1 - 1e-9)


def test_recommendation_cites_table_numbers():
    r = recommend(L8, H100, Workload(batch=32, prompt_len=1024, output_len=256), "chat")
    best = r["what_ifs"][0]
    assert best["what_if"] in r["summary"] and f"x{best['throughput_gain']:.2f}" in r["summary"]
    assert all(w["throughput_gain"] >= 0.999 for w in what_ifs(
        L8, H100, Workload(batch=32)) if w["what_if"].startswith(("HBM", "Compute")))


GRID = {"models": ["llama-3.1-8b", "llama-3.1-70b"], "hardware": ["h100-sxm", "a100-80gb"],
        "tp": [1, 3], "batch": [1, 64], "prompt_len": [512], "output_len": [64],
        "act_dtype": ["bf16", "fp8"]}


def test_sweep_skips_invalid_and_flags_capacity():
    rows, skipped = run_grid(GRID)
    assert any("tp" in s["reason"] for s in skipped)  # tp=3 splits no heads
    assert any("fp8" in s["reason"] for s in skipped)  # A100 lacks fp8
    assert any(r["fits"] is False for r in rows)  # 70B bf16 on one 80 GB GPU
    assert all(abs(sum(r[f"elasticity_{b}"] for b in ("compute", "hbm", "tier2", "network",
                                                      "overhead")) - 1) < 1e-9 for r in rows)


def test_sweep_deterministic_manifest_and_no_clobber(tmp_path):
    a = sweep(GRID, tmp_path / "a")
    b = sweep(GRID, tmp_path / "b")
    assert a["files"] == b["files"]
    manifests = [(tmp_path / x / "manifest.json").read_bytes() for x in ("a", "b")]
    assert manifests[0] == manifests[1]
    assert verify_manifest(tmp_path / "a") == []
    (tmp_path / "a/dataset.csv").write_text("tampered")
    assert verify_manifest(tmp_path / "a") == ["dataset.csv"]
    with pytest.raises(FileExistsError):
        sweep(GRID, tmp_path / "b")
    sweep(GRID, tmp_path / "b", force=True)


def test_request_rows_match_dataset(tmp_path):
    sweep({**GRID, "tp": [1], "act_dtype": ["bf16"]}, tmp_path / "d")
    first = (tmp_path / "d/dataset.csv").read_text().splitlines()[1].split(",")
    wl = Workload(batch=1, prompt_len=512, output_len=64)
    assert float(first[14]) == pytest.approx(request(L8, H100, wl)["tpot_s"], rel=1e-8)
    assert json.loads((tmp_path / "d/manifest.json").read_text())["rows"] == 8
