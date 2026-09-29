"""Intercepts (crossover points), what-if analysis, and ranked recommendations."""

from __future__ import annotations

import dataclasses

from .analyze import capacity, elasticities, request, scale_resource, step_costs
from .cost import COMPUTE, time_by_bound
from .ops import DECODE
from .specs import (
    DTYPE_BYTES,
    HardwareSpec,
    ModelSpec,
    SpecError,
    Workload,
    load_hardware,
    load_model,
)

FOUND, INCONCLUSIVE = "FOUND", "INCONCLUSIVE"
BATCH_CAP = 512  # serving systems cap concurrent sequences; a knob, not a hardware limit


def _decode_share(m, hw, wl, context, pick) -> float:
    costs = step_costs(m, hw, wl, DECODE, context)
    return pick(costs, elasticities(time_by_bound(costs, hw.overhead_mode)))


def _first_true(pred, lo: int, hi: int):
    """Smallest integer x in [lo, hi] with pred(x) true, assuming pred is monotone."""
    if pred(lo):
        return lo
    x = lo
    while x < hi and not pred(x):
        lo, x = x, min(hi, x * 2)
    if not pred(x):
        return None
    while x - lo > 1:  # invariant: pred(x) true, pred(lo) false
        mid = (lo + x) // 2
        lo, x = (lo, mid) if pred(mid) else (mid, x)
    return x


def _result(value, lo, hi, **extra) -> dict:
    return {"status": FOUND if value is not None else INCONCLUSIVE, "value": value,
            "range": [lo, hi], **extra}


def decode_knee(m: ModelSpec, hw: HardwareSpec, wl: Workload, context: int | None = None) -> dict:
    """Memory-to-compute crossovers of a decode step at KV length `context`, searched over batches
    that fit in memory. `value`: smallest batch where compute-bound time is at least half of the
    whole step. `gemm_knee`: smallest batch where weight GEMMs (and MoE experts) are mostly
    compute-bound. GQA attention stays memory-bound at any batch (its intensity does not grow with
    batch), so the whole-step knee is often INCONCLUSIVE; that is a finding, not an error."""
    c = context or wl.prompt_len
    hi = max(1, min(1 << 16, capacity(m, hw, dataclasses.replace(
        wl, prompt_len=c, output_len=1))["max_batch"]))

    def step_pred(b):
        return _decode_share(m, hw, dataclasses.replace(wl, batch=b), c,
                             lambda _, e: e[COMPUTE]) >= 0.5

    def gemm_pred(b):
        costs = step_costs(m, hw, dataclasses.replace(wl, batch=b), DECODE, c)
        g = [x for x in costs if x.op.kind in ("gemm", "moe")]
        return sum(x.op.count * x.busy[COMPUTE] for x in g if x.main == COMPUTE) >= 0.5 * sum(
            x.op.count * x.t_busy for x in g)

    ridge = (hw.peak(wl.act_dtype) * hw.compute_eff / (hw.hbm_gbps * 1e9 * hw.mem_eff)
             * DTYPE_BYTES[wl.weight_dtype] / 2)
    return _result(_first_true(step_pred, 1, hi), 1, hi, context=c,
                   gemm_knee=_first_true(gemm_pred, 1, hi),
                   unconstrained_gemm_knee=_first_true(gemm_pred, 1, 1 << 24),
                   analytic_gemm_ridge=ridge)


def attention_context(m: ModelSpec, hw: HardwareSpec, wl: Workload, hi: int = 1 << 21) -> dict:
    """Smallest KV length at which attention is at least half of decode step time."""

    def share(costs, _):
        total = sum(time_by_bound(costs, hw.overhead_mode).values())
        attn = [x for x in costs if x.op.name == "attention"]
        if hw.overhead_mode == "overlap":  # only the side on the critical path counts
            busy = sum(x.op.count * x.t_busy for x in costs)
            ovh = sum(x.op.count * x.t_overhead for x in costs)
            return sum(x.op.count * (x.t_busy if busy >= ovh else x.t_overhead)
                       for x in attn) / total
        return sum(x.time for x in attn) / total

    value = _first_true(lambda c: _decode_share(m, hw, wl, c, share) >= 0.5, 1, hi)
    return _result(value, 1, hi, batch=wl.batch)


def capacity_wall(m, hw, wl, contexts=(1024, 4096, 16384, 65536, 131072)) -> list[dict]:
    return [{"context": c, "max_batch": capacity(
        m, hw, dataclasses.replace(wl, prompt_len=c, output_len=1))["max_batch"]}
        for c in contexts]


def tp_saturation(m, hw, wl, degrees=(1, 2, 4, 8), min_gain=0.05) -> dict:
    """Smallest TP degree beyond which the next degree improves TPOT by less than `min_gain`.
    Degrees that cannot split the heads or do not fit in memory are skipped."""
    times = {}
    for tp in degrees:
        w = dataclasses.replace(wl, tp=tp)
        try:
            w.check(m)
        except SpecError:
            continue
        if capacity(m, hw, w)["fits"]:
            times[tp] = request(m, hw, w)["tpot_s"]
    ds = sorted(times)
    if not ds:
        return _result(None, degrees[0], degrees[-1], tpot_s={})
    for a, b in zip(ds, ds[1:], strict=False):
        if times[b] > times[a] * (1 - min_gain):
            return _result(a, ds[0], ds[-1], tpot_s=times)
    return _result(None, ds[0], ds[-1], tpot_s=times)


# ------------------------------------------------------------------ throughput objective


TIER2_POINTS = 16


def best_serving(m, hw, wl) -> tuple[float, float]:
    """(output tokens/s, batch) at the best real-valued batch that fits, capped at BATCH_CAP.
    Without a capacity tier throughput rises with batch, so the capacity limit is best. With one,
    batches between the HBM-only limit and the tiered limit are scanned geometrically, so adding a
    tier never lowers the optimum.
    ponytail: 16-point scan, golden-section search if the optimum's precision ever matters."""
    cap = capacity(m, hw, wl)
    b_all = min(float(BATCH_CAP), cap["max_batch_real"])
    if b_all < 1:
        return 0.0, 0.0
    candidates = [b_all]
    if hw.tier2_gb:
        kv_hbm = max(0.0, cap["hbm_budget_bytes"] - cap["weight_bytes"])
        b_hbm = max(1.0, min(b_all, kv_hbm / (cap["kv_bytes"] / wl.batch)))
        candidates += [b_hbm * (b_all / b_hbm) ** (i / TIER2_POINTS) for i in range(TIER2_POINTS)]
    best = max((request(m, hw, dataclasses.replace(wl, batch=b))["output_tokens_per_s"], b)
               for b in candidates)
    return best


def serving_batch(m, hw, wl) -> float:
    return best_serving(m, hw, wl)[1]


def max_throughput(m, hw, wl) -> float:
    """Output tokens/s at the best capacity-feasible batch (0 if one sequence does not fit)."""
    return best_serving(m, hw, wl)[0]


def throughput_elasticity(m, hw, wl, h=0.01) -> dict:
    """d ln(max throughput) / d ln(resource) by central differences, including capacity."""
    base = max_throughput(m, hw, wl)
    out = {}
    if base == 0:
        return out
    variants = {r: (scale_resource(hw, r, 1 + h), scale_resource(hw, r, 1 - h))
                for r in ("compute", "hbm", "network", "overhead")}
    variants["capacity"] = (dataclasses.replace(hw, hbm_gb=hw.hbm_gb * (1 + h)),
                            dataclasses.replace(hw, hbm_gb=hw.hbm_gb * (1 - h)))
    for r, (up, down) in variants.items():
        out[r] = (max_throughput(m, up, wl) - max_throughput(m, down, wl)) / (2 * h * base)
    return out


# ------------------------------------------------------------------ what-ifs

HW_WHATIFS = {
    "HBM bandwidth x1.5": lambda hw: dataclasses.replace(hw, hbm_gbps=hw.hbm_gbps * 1.5),
    "HBM capacity x1.5": lambda hw: dataclasses.replace(hw, hbm_gb=hw.hbm_gb * 1.5),
    "Compute x2": lambda hw: scale_resource(hw, "compute", 2.0),
    "Interconnect x2": lambda hw: scale_resource(hw, "network", 2.0),
    "Capacity tier 480 GB @ 900 GB/s": lambda hw: dataclasses.replace(
        hw, tier2_gb=480.0, tier2_gbps=900.0),
    "Capacity tier 480 GB @ 64 GB/s": lambda hw: dataclasses.replace(
        hw, tier2_gb=480.0, tier2_gbps=64.0),
}
SW_WHATIFS = {
    "KV cache fp8": lambda wl: dataclasses.replace(wl, kv_dtype="fp8"),
    "Weights fp8": lambda wl: dataclasses.replace(wl, weight_dtype="fp8"),
}


def what_ifs(m, hw, wl) -> list[dict]:
    """Latency speedup at the workload's batch and throughput gain at the capacity-limited batch
    for each what-if, sorted by throughput gain then latency speedup."""
    base_lat = request(m, hw, wl)["e2e_s"] if capacity(m, hw, wl)["fits"] else None
    base_tput = max_throughput(m, hw, wl)
    rows = []
    variants = [(n, f(hw), wl) for n, f in HW_WHATIFS.items()]
    variants += [(n, hw, f(wl)) for n, f in SW_WHATIFS.items()]
    for name, h2, w2 in variants:
        if h2.link_gbps == 0 and w2.tp > 1:
            continue
        lat = request(m, h2, w2)["e2e_s"] if capacity(m, h2, w2)["fits"] else None
        tput = max_throughput(m, h2, w2)
        rows.append({
            "what_if": name,
            "latency_speedup": base_lat / lat if base_lat and lat else None,
            "throughput_gain": tput / base_tput if base_tput else None,
            "serving_batch": serving_batch(m, h2, w2),
        })
    rows.sort(key=lambda r: (-(r["throughput_gain"] or 0), -(r["latency_speedup"] or 0),
                             r["what_if"]))
    return rows


def tier2_breakeven(m, hw, wl, tier2_gb=480.0) -> dict:
    """Capacity-tier bandwidth (GB/s) above which spilling KV to a tier of `tier2_gb` raises
    capacity-limited throughput over HBM alone."""
    base = max_throughput(m, hw, wl)

    def better(gbps):
        t = dataclasses.replace(hw, tier2_gb=tier2_gb, tier2_gbps=float(gbps))
        return max_throughput(m, t, wl) > base * 1.001

    hi = int(hw.hbm_gbps)
    value = _first_true(better, 1, hi) if base > 0 else None
    return _result(value, 1, hi, tier2_gb=tier2_gb, base_tokens_per_s=base)


def recommend(m, hw, wl, archetype: str = "") -> dict:
    """Deterministic recommendation text rendered from the what-if table and elasticities."""
    table = what_ifs(m, hw, wl)
    fits = capacity(m, hw, wl)["fits"]
    el = request(m, hw, wl)["elasticity"] if fits else {}
    tel = throughput_elasticity(m, hw, wl)
    lines = []
    if el:
        top = max(el, key=el.get)
        lines.append(f"At batch {wl.batch:g}, {el[top]:.0%} of request time is {top}-bound "
                     "(latency elasticity).")
    else:
        lines.append(f"Batch {wl.batch:g} does not fit in memory; latency is undefined.")
    if tel:
        top_t = max(tel, key=tel.get)
        lines.append(f"Capacity-limited throughput is most sensitive to {top_t} "
                     f"(elasticity {tel[top_t]:.2f}).")
    best = table[0] if table and table[0]["throughput_gain"] else None
    if best:
        lines.append(f"Largest throughput intercept: {best['what_if']} "
                     f"(x{best['throughput_gain']:.2f} tokens/s).")
    return {"archetype": archetype, "model": m.name, "hardware": hw.name,
            "workload": dataclasses.asdict(wl), "latency_elasticity": el,
            "throughput_elasticity": tel, "what_ifs": table, "summary": " ".join(lines)}


def study_cases(study: dict) -> list[tuple[str, ModelSpec, HardwareSpec, Workload]]:
    """Archetype x focus cases of a study, in file order."""
    cases = []
    for f in study["focus"]:
        m, hw = load_model(f["model"]), load_hardware(f["hardware"])
        extra = {k: v for k, v in f.items() if k not in ("model", "hardware")}
        for a in study["archetypes"]:
            wl = Workload(batch=a["batch"], prompt_len=a["prompt_len"],
                          output_len=a["output_len"], **extra)
            wl.check(m)
            cases.append((a["name"], m, hw, wl))
    return cases
