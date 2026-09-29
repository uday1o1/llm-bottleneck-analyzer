"""Phase and request analysis: TTFT, TPOT, capacity, critical path, and resource elasticities."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from . import __version__
from .cost import BOUNDS, OpCost, cost_ops, time_by_bound
from .ops import (
    DECODE,
    PREFILL,
    build_ops,
    kv_bytes_per_token,
    param_counts,
    weight_bytes_per_rank,
)
from .specs import HardwareSpec, ModelSpec, Workload, spec_hash, to_dict

TPOT_SAMPLES = 64


def step_costs(m: ModelSpec, hw: HardwareSpec, wl: Workload, phase: str, context: int = 0):
    frac = capacity(m, hw, wl)["tier2_frac"] if phase == DECODE and hw.tier2_gb else 0.0
    return cost_ops(build_ops(m, wl, phase, context, frac), hw, wl.act_dtype)


def decode_positions(wl: Workload) -> list[float]:
    """KV lengths at which decode steps are evaluated. Exact for short outputs; otherwise the
    midpoint rule over TPOT_SAMPLES strata.
    ponytail: sampled mean, exact sum if a bound transition inside a stratum ever matters."""
    S, O = wl.prompt_len, wl.output_len
    if O <= TPOT_SAMPLES:
        return [S + i + 1 for i in range(O)]
    return [S + (j + 0.5) * O / TPOT_SAMPLES + 0.5 for j in range(TPOT_SAMPLES)]


@dataclass
class PhaseResult:
    time: float
    by_bound: dict[str, float]
    by_op: dict[str, float]
    costs: list[OpCost]  # representative step (prefill pass, or the median decode position)


def _phase(m, hw, wl, phase) -> PhaseResult:
    if phase == PREFILL:
        steps = [step_costs(m, hw, wl, PREFILL)]
    else:
        steps = [step_costs(m, hw, wl, DECODE, round(c)) for c in decode_positions(wl)]
    n = len(steps)
    by_bound = dict.fromkeys(BOUNDS, 0.0)
    by_op: dict[str, float] = {}
    for costs in steps:
        tb = time_by_bound(costs, hw.overhead_mode)
        step = sum(tb.values())
        raw = {c.op.name: c.time for c in costs}
        scale = step / sum(raw.values())  # 1.0 in additive mode
        for k, v in tb.items():
            by_bound[k] += v / n
        for k, v in raw.items():
            by_op[k] = by_op.get(k, 0.0) + v * scale / n
    return PhaseResult(sum(by_bound.values()), by_bound, by_op, steps[n // 2])


def capacity(m: ModelSpec, hw: HardwareSpec, wl: Workload) -> dict:
    """Memory budget per rank. Weights live in HBM (activation workspace is folded into
    `capacity_usable`); KV fills the remaining HBM, then spills to the capacity tier. Each
    sequence is sized at its final length S+O (static batch)."""
    hbm = hw.hbm_gb * 1e9 * hw.capacity_usable
    tier2 = hw.tier2_gb * 1e9
    weights = weight_bytes_per_rank(m, wl)
    per_seq = (wl.prompt_len + wl.output_len) * kv_bytes_per_token(m, wl)
    kv = wl.batch * per_seq
    kv_hbm = max(0.0, hbm - weights)
    return {
        "hbm_budget_bytes": hbm,
        "tier2_bytes": tier2,
        "weight_bytes": weights,
        "kv_bytes_per_token": kv_bytes_per_token(m, wl),
        "kv_bytes": kv,
        "max_batch_real": (kv_hbm + tier2) / per_seq if weights <= hbm else 0.0,
        "max_batch": int((kv_hbm + tier2) // per_seq) if weights <= hbm else 0,
        "max_batch_hbm_only": int(kv_hbm // per_seq),
        "fits": weights <= hbm and kv <= kv_hbm + tier2,
        "tier2_frac": max(0.0, 1 - kv_hbm / kv) if kv else 0.0,
    }


def elasticities(by_bound: dict[str, float]) -> dict[str, float]:
    """-dlnT/dln(speedup) for each resource, which equals its critical-path time share."""
    total = sum(by_bound.values())
    return {k: v / total for k, v in by_bound.items()} if total else dict.fromkeys(by_bound, 0.0)


def request(m: ModelSpec, hw: HardwareSpec, wl: Workload) -> dict:
    """Static-batch request metrics."""
    wl.check(m)
    pre, dec = _phase(m, hw, wl, PREFILL), _phase(m, hw, wl, DECODE)
    O = wl.output_len
    e2e = pre.time + O * dec.time
    total_by_bound = {k: pre.by_bound[k] + O * dec.by_bound[k] for k in BOUNDS}
    return {
        "ttft_s": pre.time,
        "tpot_s": dec.time,
        "e2e_s": e2e,
        "decode_tokens_per_s": wl.batch / dec.time,
        "output_tokens_per_s": wl.batch * O / e2e,
        "prefill": pre,
        "decode": dec,
        "elasticity": elasticities(total_by_bound),
        "elasticity_prefill": elasticities(pre.by_bound),
        "elasticity_decode": elasticities(dec.by_bound),
    }


def critical_path(phase: PhaseResult, top: int = 8) -> list[dict]:
    """Ops ranked by share of phase time, with their binding resource."""
    bound_of = {c.op.name: c.bound for c in phase.costs}
    rows = sorted(phase.by_op.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    return [{"op": k, "time_s": v, "share": v / phase.time, "bound": bound_of[k]} for k, v in rows]


def op_table(costs: list[OpCost]) -> list[dict]:
    rows = []
    for c in costs:
        o = c.op
        rows.append({
            "op": o.name, "kind": o.kind, "count": o.count, "flops": o.flops,
            "hbm_bytes": o.hbm_bytes, "tier2_bytes": o.tier2_bytes,
            "weight_bytes": o.weight_bytes, "kv_bytes": o.kv_bytes,
            "intensity": o.flops / o.hbm_bytes if o.hbm_bytes else None,
            "time_s": c.time, "bound": c.bound,
        })
    return rows


def analyze(m: ModelSpec, hw: HardwareSpec, wl: Workload) -> dict:
    """Full JSON-serializable characterization of one operating point."""
    r = request(m, hw, wl)
    total, active = param_counts(m)
    return {
        "llmba_version": __version__,
        "model": m.name, "hardware": hw.name, "workload": to_dict(wl),
        "spec_hashes": {"model": spec_hash(m), "hardware": spec_hash(hw)},
        "params": {"total": total, "active": active},
        "capacity": capacity(m, hw, wl),
        "ridge_intensity": hw.peak(wl.act_dtype) * hw.compute_eff
        / (hw.hbm_gbps * 1e9 * hw.mem_eff),
        "metrics": {k: r[k] for k in ("ttft_s", "tpot_s", "e2e_s", "decode_tokens_per_s",
                                      "output_tokens_per_s")},
        "elasticity": {"request": r["elasticity"], "prefill": r["elasticity_prefill"],
                       "decode": r["elasticity_decode"]},
        "critical_path": {"prefill": critical_path(r["prefill"]),
                          "decode": critical_path(r["decode"])},
        "ops": {"prefill": op_table(r["prefill"].costs), "decode": op_table(r["decode"].costs)},
        "unverified_hw_fields": list(hw.unverified),
    }


def throughput_ceiling(m: ModelSpec, hw: HardwareSpec, prompt_len: int, output_len: int,
                       n_gpus: int) -> dict:
    """Scheduler-independent upper bound on aggregate output tokens/s for a dense model (S3).

    Every decode step reads all weights W once plus the KV of each resident sequence, and resident
    KV never exceeds Cap - W. Summed over steps: steps >= N * KV_req / (Cap - W), so each request
    moves at least M_req = KV_req * (1 + W_read / (Cap - W)) bytes (W_read excludes the gathered
    input embedding), with
    KV_req = kv_tok * (O-1) * (S + (O-2)/2). Then t_req >= max(F_req / P, M_req / BW) and
    throughput <= O / t_req. Capacity is binary GiB, all FLOPs run at the best dense peak,
    weights and KV are 1 byte, efficiencies are 1, overhead is 0: each choice only raises the
    ceiling, so a measured throughput above it falsifies the model."""
    if m.moe:
        raise ValueError("throughput_ceiling: dense models only (MoE steps read a subset of W)")
    S, O = prompt_len, output_len
    one = Workload(batch=1, prompt_len=S, output_len=O, weight_dtype="fp8", kv_dtype="fp8",
                   act_dtype="fp8")
    flops = sum(o.flops * o.count for o in build_ops(m, one, PREFILL))
    for i in range(1, O):  # decode steps after the prefill-produced first token
        flops += sum(o.flops * o.count for o in build_ops(m, one, DECODE, S + i))
    kv_req = kv_bytes_per_token(m, one) * (O - 1) * (S + (O - 2) / 2)
    w_res = weight_bytes_per_rank(m, one)
    w_read = w_res - m.vocab * m.d_model  # the input embedding is gathered, not streamed
    cap = n_gpus * hw.hbm_gb * 2**30
    if w_res >= cap:
        return {"tokens_per_s": 0.0, "fits": False}
    m_req = kv_req * (1 + w_read / (cap - w_res))
    peak = n_gpus * max(hw.peak_tflops.values()) * 1e12
    bw = n_gpus * hw.hbm_gbps * 1e9
    t_req = max(flops / peak, m_req / bw)
    return {"tokens_per_s": O / t_req, "t_req_s": t_req, "fits": True,
            "flops_per_req": flops, "bytes_per_req": m_req,
            "bound": "compute" if flops / peak >= m_req / bw else "hbm"}


RESOURCES = BOUNDS


def scale_resource(hw: HardwareSpec, resource: str, factor: float) -> HardwareSpec:
    """Make one resource `factor` times faster. Network scales link bandwidth up and link latency
    down together; overhead scales kernel launch cost down."""
    if resource == "compute":
        peaks = {k: v * factor for k, v in hw.peak_tflops.items()}
        return dataclasses.replace(hw, peak_tflops=peaks)
    if resource == "hbm":
        return dataclasses.replace(hw, hbm_gbps=hw.hbm_gbps * factor)
    if resource == "tier2":
        return dataclasses.replace(hw, tier2_gbps=hw.tier2_gbps * factor)
    if resource == "network":
        return dataclasses.replace(hw, link_gbps=hw.link_gbps * factor,
                                   link_latency_us=hw.link_latency_us / factor)
    if resource == "overhead":
        return dataclasses.replace(hw, kernel_overhead_us=hw.kernel_overhead_us / factor)
    raise ValueError(f"resource: {resource!r}")


def check_published(ref: dict, load_model, load_hardware) -> dict:
    """Claim S3: every published measured throughput must be at most the ceiling."""
    rows = []
    for r in ref["rows"]:
        c = throughput_ceiling(load_model(r["model"]), load_hardware(r["hardware"]),
                               r["prompt_len"], r["output_len"], r["n_gpus"])
        rows.append({**r, "ceiling_tokens_per_s": c["tokens_per_s"], "ceiling_bound": c["bound"],
                     "ratio": r["output_tokens_per_s"] / c["tokens_per_s"]})
    ratios = sorted(x["ratio"] for x in rows)
    return {"source_url": ref["source_url"], "rows": rows, "n": len(rows),
            "min_ratio": ratios[0], "median_ratio": ratios[len(ratios) // 2],
            "max_ratio": ratios[-1], "verdict": "PASS" if ratios[-1] <= 1.0 else "FAIL"}
