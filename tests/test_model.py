import dataclasses
import math
import random

import pytest

from llmba.analyze import (
    RESOURCES,
    analyze,
    capacity,
    decode_positions,
    request,
    scale_resource,
    step_costs,
    throughput_ceiling,
)
from llmba.cost import BOUNDS
from llmba.ops import (
    DECODE,
    PREFILL,
    build_ops,
    expected_distinct_experts,
    kv_bytes_per_token,
    param_counts,
    weight_bytes_per_rank,
)
from llmba.specs import Workload, load_hardware, load_model, preset_names

L8 = load_model("llama-3.1-8b")
H100 = load_hardware("h100-sxm")


def test_kv_bytes_per_token_golden():
    assert kv_bytes_per_token(L8, Workload()) == 131072  # 32 layers * 2 * 8 heads * 128 * 2 B
    assert kv_bytes_per_token(load_model("llama-2-7b"), Workload()) == 524288
    ds = load_model("deepseek-v3")
    assert kv_bytes_per_token(ds, Workload()) == 61 * (512 + 64) * 2
    assert kv_bytes_per_token(ds, Workload(tp=8)) == 61 * (512 + 64) * 2  # latent replicated
    assert kv_bytes_per_token(L8, Workload(tp=4)) == 131072 / 4


@pytest.mark.parametrize("name", preset_names("models"))
def test_decode_weight_bytes_equal_active_params(name):
    """At one token, every op reads each active weight once (MoE touches exactly top_k experts)."""
    m = load_model(name)
    wl = Workload(batch=1, weight_dtype="bf16")
    ops = build_ops(m, wl, DECODE, context=1)
    weights = sum(o.weight_bytes * o.count for o in ops) / 2
    _, active = param_counts(m)
    embed = m.vocab * m.d_model  # gathered row, not streamed
    norms = (2 * m.n_layers + 1) * m.d_model
    mla_norms = m.n_layers * (m.mla.q_lora_rank + m.mla.kv_lora_rank) if m.mla else 0
    assert weights == pytest.approx(active - embed - norms - mla_norms, rel=1e-9)


def test_dense_decode_flops_are_two_per_param():
    ops = build_ops(L8, Workload(batch=1), DECODE, context=1)
    gemm_flops = sum(o.flops * o.count for o in ops if o.kind == "gemm")
    _, active = param_counts(L8)
    assert gemm_flops == pytest.approx(2 * (active - L8.vocab * L8.d_model), rel=0.001)


def test_prefill_attention_flops_closed_form():
    wl = Workload(batch=2, prompt_len=100)
    attn = next(o for o in build_ops(L8, wl, PREFILL) if o.name == "attention")
    assert attn.flops == 4 * 2 * 32 * 128 * 100 * 101 / 2


def test_expected_distinct_experts():
    assert expected_distinct_experts(8, 2, 1) == pytest.approx(2)
    assert expected_distinct_experts(256, 8, 10_000) == pytest.approx(256)
    assert expected_distinct_experts(256, 8, 4) < 32  # collisions


def e2e(m, hw, wl):
    return request(m, hw, wl)["e2e_s"]


@pytest.mark.parametrize("name", preset_names("models"))
def test_monotone_in_batch_and_context(name):
    m = load_model(name)
    hw = load_hardware("b200")
    base = Workload(batch=4, prompt_len=512, output_len=32, tp=8 if m.mla else 1)
    t = e2e(m, hw, base)
    assert e2e(m, hw, dataclasses.replace(base, batch=8)) >= t
    assert e2e(m, hw, dataclasses.replace(base, prompt_len=1024)) >= t
    for r in RESOURCES:
        assert e2e(m, scale_resource(hw, r, 2.0), base) <= t


def test_memory_bound_decode_scales_with_bandwidth():
    hw = dataclasses.replace(H100, kernel_overhead_us=0.0)
    wl = Workload(batch=1, prompt_len=128, output_len=8)
    t1 = request(L8, hw, wl)["tpot_s"]
    t2 = request(L8, scale_resource(hw, "hbm", 2.0), wl)["tpot_s"]
    assert t2 == pytest.approx(t1 / 2, rel=1e-3)


def test_scaling_every_rate_scales_time():
    hw = dataclasses.replace(H100, kernel_overhead_us=0.0, link_latency_us=0.0)
    fast = hw
    for r in ("compute", "hbm", "network"):
        fast = scale_resource(fast, r, 2.0)
    wl = Workload(batch=16, prompt_len=512, output_len=16, tp=2)
    assert e2e(L8, fast, wl) == pytest.approx(e2e(L8, hw, wl) / 2, rel=1e-9)


def _near_tie(m, hw, wl) -> bool:
    for phase, ctxs in ((PREFILL, [0]), (DECODE, [round(c) for c in decode_positions(wl)])):
        for c in ctxs:
            for cost in step_costs(m, hw, wl, phase, c):
                parts = sorted(cost.busy.values())
                if parts[2] > 0 and parts[1] / parts[2] > 0.99:
                    return True
    return False


def test_elasticity_equals_finite_difference():
    """Claim S2: analytic elasticity (time share) matches central finite differences."""
    rng = random.Random(0)
    models = preset_names("models")
    hws = [h for h in preset_names("hardware")]
    checked = skipped = 0
    h = 1e-3
    for _ in range(200):
        m = load_model(rng.choice(models))
        hw = load_hardware(rng.choice(hws))
        if rng.random() < 0.3:
            hw = dataclasses.replace(hw, tier2_gb=480, tier2_gbps=rng.choice([100, 900]))
        if rng.random() < 0.3:
            hw = dataclasses.replace(hw, overhead_mode="overlap")
        tp = rng.choice([1, 2, 4, 8])
        wl = Workload(batch=rng.choice([1, 4, 32, 128]), prompt_len=rng.choice([128, 2048, 8192]),
                      output_len=rng.choice([16, 200]), tp=tp)
        if _near_tie(m, hw, wl):
            skipped += 1
            continue
        el = request(m, hw, wl)["elasticity"]
        for r in RESOURCES:
            up = e2e(m, scale_resource(hw, r, 1 + h), wl)
            down = e2e(m, scale_resource(hw, r, 1 - h), wl)
            fd = -(math.log(up) - math.log(down)) / (math.log(1 + h) - math.log(1 - h))
            assert fd == pytest.approx(el[r], abs=0.02), (m.name, hw.name, wl, r)
        checked += 1
    assert checked >= 100, (checked, skipped)


def test_tpot_sampling_close_to_exact():
    wl = Workload(batch=64, prompt_len=100, output_len=4000)
    sampled = request(L8, H100, wl)["tpot_s"]
    exact = sum(
        sum(c.time for c in step_costs(L8, H100, wl, DECODE, 100 + i + 1)) for i in range(4000)
    ) / 4000
    assert sampled == pytest.approx(exact, rel=0.005)


def test_capacity():
    wl = Workload(batch=1, prompt_len=4096, output_len=4096)
    cap = capacity(L8, H100, wl)
    expected = int((80e9 * 0.9 - 8.030261248e9 * 2) // (8192 * 131072))
    assert cap["max_batch"] == expected and cap["fits"] and cap["tier2_frac"] == 0
    assert not capacity(load_model("llama-3.1-70b"), H100, wl)["fits"]  # 141 GB of weights


def test_tp_adds_network_ops_and_time():
    wl = Workload(batch=8, prompt_len=256, output_len=8, tp=4)
    r = analyze(load_model("llama-3.1-70b"), H100, wl)
    assert r["elasticity"]["request"]["network"] > 0
    assert set(r["elasticity"]["request"]) == set(BOUNDS)
    assert any(o["op"] == "attn_allreduce" for o in r["ops"]["decode"])


def test_mla_decode_attention_formula():
    ds = load_model("deepseek-v3")
    wl = Workload(batch=4, tp=8)
    attn = next(o for o in build_ops(ds, wl, DECODE, context=1000) if o.name == "attention")
    h, r, dr = 128 // 8, 512, 64
    absorb = 2 * 4 * r * h * (128 + 128)
    assert attn.flops == pytest.approx(2 * 4 * 1000 * h * (2 * r + dr) + absorb)
    assert attn.kv_bytes == 4 * 1000 * (r + dr) * 2 + 4 * (r + dr) * 2  # not divided by tp


def test_tier2_spill_extends_capacity_and_costs_bandwidth():
    wl = Workload(batch=64, prompt_len=16384, output_len=512)
    assert not capacity(L8, H100, wl)["fits"]
    tiered = dataclasses.replace(H100, tier2_gb=480, tier2_gbps=900)
    cap = capacity(L8, tiered, wl)
    assert cap["fits"] and 0 < cap["tier2_frac"] < 1
    slow = dataclasses.replace(tiered, tier2_gbps=64)
    assert request(L8, slow, wl)["tpot_s"] > request(L8, tiered, wl)["tpot_s"]
    assert request(L8, slow, wl)["elasticity"]["tier2"] > 0.3


def test_overlap_mode_hides_overhead_when_busy_dominates():
    wl = Workload(batch=1, prompt_len=128, output_len=4)
    add = request(L8, H100, wl)
    ovl = request(L8, dataclasses.replace(H100, overhead_mode="overlap"), wl)
    assert ovl["tpot_s"] < add["tpot_s"]
    assert ovl["elasticity_decode"]["overhead"] == 0
    tiny = dataclasses.replace(H100, overhead_mode="overlap", kernel_overhead_us=1e4)
    assert request(L8, tiny, wl)["elasticity_decode"]["overhead"] == 1.0


def test_throughput_ceiling_closed_form():
    """Memory term equals KV_req * (1 + W_read / (Cap - W)) with fp8 weights and KV."""
    c = throughput_ceiling(L8, H100, prompt_len=1000, output_len=1000, n_gpus=1)
    kv_tok = 131072 / 2
    kv_req = kv_tok * 999 * (1000 + 998 / 2)
    cap = 80 * 2**30
    w = weight_bytes_per_rank(L8, Workload(weight_dtype="fp8"))
    assert c["bytes_per_req"] == pytest.approx(kv_req * (1 + (w - 128256 * 4096) / (cap - w)))
    two = throughput_ceiling(L8, H100, 1000, 1000, n_gpus=2)
    assert two["tokens_per_s"] > c["tokens_per_s"]
    with pytest.raises(ValueError):
        throughput_ceiling(load_model("mixtral-8x7b"), H100, 128, 128, 1)


def test_stored_weights_replicate_kv_heads_beyond_n_kv_heads():
    m = load_model("llama-3.1-70b")
    for tp in (8, 16, 64):
        wl = Workload(batch=1, tp=tp)
        streamed = sum(o.weight_bytes * o.count for o in build_ops(m, wl, DECODE, 1))
        embed = m.vocab * m.d_model / tp * 2
        norms = (2 * m.n_layers + 1) * m.d_model * 2
        assert weight_bytes_per_rank(m, wl) == pytest.approx(streamed + embed + norms, rel=1e-9)
