"""Closed-loop empirical check (claim P1): calibrate a device from microbenchmarks, then predict and
measure an executable dense transformer twin's decode step, and grade the prediction.

Calibration never sees twin measurements; efficiency knobs are never fitted to them."""

from __future__ import annotations

import dataclasses
import json
import platform
import random
import statistics
import time
from pathlib import Path

from . import __version__
from .analyze import step_costs
from .cost import HBM, time_by_bound
from .ops import DECODE
from .specs import (
    HardwareSpec,
    SpecError,
    Twin,
    Workload,
    load_hardware,
    load_twin,
    parse_sets,
    spec_hash,
    to_dict,
    with_overrides,
)
from .sweep import dumps, prepare_dir, write_files

WARMUP, TRIALS, CV_LIMIT, SUPPORT_FLOOR = 3, 15, 0.25, 0.8
MDAPE_LIMIT, BOOT_N, BOOT_SEED, KNEE_RATIO = 0.30, 2000, 0, 1.5
PASS, FAIL, INSUFFICIENT, BLOCKED = "PASS", "FAIL", "INSUFFICIENT_EVIDENCE", "BLOCKED"


# ------------------------------------------------------------------ pure grading logic


def bootstrap_mdape(ape: list[float], n: int = BOOT_N, seed: int = BOOT_SEED) -> tuple:
    """(MdAPE, 2.5th, 97.5th percentile) by resampling grid points with replacement."""
    rng = random.Random(seed)
    meds = sorted(statistics.median(rng.choices(ape, k=len(ape))) for _ in range(n))
    return statistics.median(ape), meds[int(0.025 * n)], meds[int(0.975 * n) - 1]


def knee(points: dict[int, float]) -> int | None:
    """Smallest batch whose step time is at least KNEE_RATIO x the batch-1 time."""
    base = points.get(1)
    if base is None:
        return None
    return next((b for b in sorted(points) if points[b] >= KNEE_RATIO * base), None)


def knee_agrees(pred: int | None, meas: int | None) -> bool:
    """Within one power-of-two grid step; both None also agrees."""
    if pred is None or meas is None:
        return pred is None and meas is None
    return max(pred, meas) <= 2 * min(pred, meas)


def grade(rows: list[dict], total_points: int) -> dict:
    """Verdict for claim P1 from per-point rows with `valid`, `measured_s`, `predicted_s`."""
    valid = [r for r in rows if r["valid"]]
    out = {"points": total_points, "valid_points": len(valid)}
    if len(valid) < SUPPORT_FLOOR * total_points or len(valid) < 2:
        return {**out, "verdict": INSUFFICIENT}
    ape = [abs(r["predicted_s"] - r["measured_s"]) / r["measured_s"] for r in valid]
    md, lo, hi = bootstrap_mdape(ape)
    c0 = min(r["context"] for r in rows)  # the grid's shortest context, valid or not
    base_ok = any(r["valid"] and r["batch"] == 1 and r["context"] == c0 for r in rows)
    k_meas = knee({r["batch"]: r["measured_s"] for r in valid if r["context"] == c0})
    k_pred = knee({r["batch"]: r["predicted_s"] for r in valid if r["context"] == c0})
    agrees = base_ok and knee_agrees(k_pred, k_meas)  # an unresolvable knee never agrees
    ok = hi <= MDAPE_LIMIT and agrees
    return {**out, "mdape": md, "mdape_ci95": [lo, hi], "knee_context": c0,
            "knee_resolvable": base_ok, "knee_measured": k_meas, "knee_predicted": k_pred,
            "knee_agrees": agrees, "verdict": PASS if ok else FAIL}


def grade_negative_control(rows: list[dict]) -> dict:
    """The x4-bandwidth control must be rejected on MdAPE alone: CI lower bound above the limit."""
    valid = [r for r in rows if r["valid"]]
    ape = [abs(r["control_s"] - r["measured_s"]) / r["measured_s"] for r in valid]
    if not ape:
        return {"mdape": None, "mdape_ci95": [None, None], "rejected": False}
    md, lo, hi = bootstrap_mdape(ape)
    return {"mdape": md, "mdape_ci95": [lo, hi], "rejected": lo > MDAPE_LIMIT}


# ------------------------------------------------------------------ prediction


def twin_workload(tw: Twin, batch: int) -> Workload:
    return Workload(batch=batch, prompt_len=1, output_len=1, weight_dtype=tw.dtype,
                    kv_dtype=tw.dtype, act_dtype=tw.dtype)


def predict(tw: Twin, hw: HardwareSpec, batch: int, context: int) -> tuple[float, float]:
    """(predicted decode-step seconds, HBM share of the step)."""
    tb = time_by_bound(step_costs(tw.model, hw, twin_workload(tw, batch), DECODE, context),
                       hw.overhead_mode)
    total = sum(tb.values())
    return total, tb[HBM] / total


def grid(tw: Twin) -> list[tuple[int, int]]:
    return [(b, c) for c in tw.contexts for b in tw.batches if b * c <= tw.max_kv_tokens]


def grid_check(tw: Twin, hw: HardwareSpec) -> dict:
    """Pre-registration rule: at least 32 points, at least two thirds with HBM share >= 0.7."""
    pts = grid(tw)
    mem = sum(predict(tw, hw, b, c)[1] >= 0.7 for b, c in pts)
    pow2 = all(b & (b - 1) == 0 for b in tw.batches)
    return {"points": len(pts), "hbm_share_ge_0.7": mem, "power_of_two_batches": pow2,
            "ok": len(pts) >= 32 and mem >= 2 * len(pts) / 3 and pow2}


# ------------------------------------------------------------------ torch side


def _torch():
    try:
        import torch
    except ImportError as e:
        raise SpecError("measure: torch is required: pip install '.[measure]'") from e
    return torch


def pick_device(requested: str):
    torch = _torch()
    avail = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(),
             "cpu": True}
    if requested not in (*avail, "auto"):
        raise SpecError(f"device: unknown device {requested!r}")
    if requested == "auto":
        requested = next(d for d in ("cuda", "mps", "cpu") if avail[d])
    return requested if avail[requested] else None


def _sync(device):
    torch = _torch()
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def time_fn(fn, device, trials=TRIALS, warmup=WARMUP) -> list[float]:
    for _ in range(warmup):
        fn()
    _sync(device)
    out = []
    for _ in range(trials):
        t = time.perf_counter()
        fn()
        _sync(device)
        out.append(time.perf_counter() - t)
    return out


LOOP_REPS = 20
GEMM_SHAPES = ((4096, 4096), (4096, 11008))


def time_loop(fn, device, reps=LOOP_REPS, trials=TRIALS) -> list[float]:
    """Per-call seconds from `reps` back-to-back calls per trial."""
    return [t / reps for t in time_fn(lambda: [fn() for _ in range(reps)], device, trials)]


def _cv(xs):
    return statistics.stdev(xs) / statistics.mean(xs)


def calibrate(device: str) -> tuple[HardwareSpec, dict]:
    """Copy bandwidth (read + write bytes), large-GEMM throughput, per-launch overhead, matmul
    efficiency by row count, and KV-streaming attention efficiency. Standalone microbenchmarks
    only: nothing here sees a twin."""
    torch = _torch()
    dt = torch.float16
    a = torch.ones(128 * 2**20, dtype=dt, device=device)  # 256 MiB
    b = torch.empty_like(a)
    copy = time_fn(lambda a=a, b=b: b.copy_(a), device)
    gbps = 2 * a.numel() * a.element_size() / statistics.median(copy) / 1e9
    del a, b
    n = 4096
    x = torch.randn(n, n, dtype=dt, device=device)
    y = torch.randn(n, n, dtype=dt, device=device)
    gemm = time_fn(lambda x=x, y=y: x @ y, device, trials=10)
    tflops = 2 * n**3 / statistics.median(gemm) / 1e12
    del x, y
    z = torch.zeros(1, dtype=dt, device=device)
    launches = 200

    def tiny():
        for _ in range(launches):
            z.add_(1)

    tiny_t = time_fn(tiny, device, trials=10)
    overhead_us = statistics.median(tiny_t) / launches * 1e6
    # Operator shape efficiency (attained / roofline time), timed as back-to-back loops so a
    # device sync is not charged to each sub-millisecond call. Generic shapes, not twin shapes.
    shape_t, eff = {}, {}
    for k, nn in GEMM_SHAPES:
        w = torch.randn(k, nn, dtype=dt, device=device)
        for m_rows in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            xm = torch.randn(m_rows, k, dtype=dt, device=device)
            ts = time_loop(lambda xm=xm, w=w: xm @ w, device)
            roof = max(2 * m_rows * k * nn / (tflops * 1e12),
                       2 * (k * nn + m_rows * (k + nn)) / (gbps * 1e9))
            shape_t[f"{k}x{nn}/{m_rows}"] = ts
            eff.setdefault(str(m_rows), []).append(roof / statistics.median(ts))
        del w
    gemm_eff = {m: min(2.0, statistics.mean(v)) for m, v in eff.items()}
    attn_t, attn_ratios = {}, []
    hkv, g, hd = 8, 4, 128
    for bsz, ctx in ((1, 1024), (1, 4096), (1, 16384), (4, 1024), (4, 4096)):
        q = torch.randn(bsz, hkv, g, hd, dtype=dt, device=device)
        kt = torch.randn(bsz, hkv, hd, ctx, dtype=dt, device=device)
        v = torch.randn(bsz, hkv, ctx, hd, dtype=dt, device=device)
        ts = time_loop(lambda q=q, kt=kt, v=v: torch.softmax(q @ kt, dim=-1) @ v, device)
        attn_t[f"{bsz}/{ctx}"] = ts
        attn_ratios.append(2 * kt.numel() * 2 / (gbps * 1e9) / statistics.median(ts))
        del q, kt, v
    attn_eff = min(2.0, statistics.median(attn_ratios))
    mem_gb = (torch.mps.recommended_max_memory() / 1e9 if device == "mps"
              else torch.cuda.get_device_properties(0).total_memory / 1e9 if device == "cuda"
              else 8.0)
    hw = HardwareSpec(
        name=f"local-{device}", peak_tflops={"fp16": tflops}, hbm_gb=mem_gb, hbm_gbps=gbps,
        compute_eff=1.0, mem_eff=1.0, kernel_overhead_us=overhead_us, capacity_usable=0.9,
        gemm_eff_by_tokens=gemm_eff, attn_eff=attn_eff,
        memory_type="measured", notes="Calibrated from microbenchmarks (achieved, not peak).")
    evidence = {"device": device, "device_name": _device_name(device),
                "torch": torch.__version__, "platform": platform.platform(),
                "python": platform.python_version(), "llmba_version": __version__,
                "copy_s": copy, "gemm_s": gemm, "tiny_launch_loop_s": tiny_t,
                "launches_per_loop": launches, "gemm_n": n, "shape_sweep_s": shape_t,
                "attention_sweep_s": attn_t, "loop_reps": LOOP_REPS,
                "cv": {"copy": _cv(copy), "gemm": _cv(gemm), "tiny": _cv(tiny_t)}}
    return hw, evidence


def _device_name(device):
    torch = _torch()
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    if device == "mps":
        return f"Apple GPU ({platform.machine()}, {platform.processor() or 'arm'})"
    return platform.processor() or platform.machine()


class TwinModel:
    """Eager PyTorch dense GQA decoder with random weights (SwiGLU MLP, RMSNorm, no RoPE)."""

    def __init__(self, tw: Twin, device: str, seed: int = 0):
        torch = _torch()
        m = tw.model
        if m.moe or m.mla:
            raise SpecError("twin: only dense GQA twins are executable (MoE/MLA are follow-on)")
        g = torch.Generator(device="cpu").manual_seed(seed)
        dt = getattr(torch, {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}[tw.dtype])
        d, hd, h, hkv = m.d_model, m.head_dim, m.n_heads, m.n_kv_heads

        def w(*shape):
            return (torch.randn(*shape, generator=g) / shape[0] ** 0.5).to(device=device, dtype=dt)

        self.m, self.dt, self.device = m, dt, device
        self.embed = w(m.vocab, d)
        self.layers = [{"n1": torch.ones(d, dtype=dt, device=device),
                        "n2": torch.ones(d, dtype=dt, device=device),
                        "qkv": w(d, (h + 2 * hkv) * hd), "o": w(h * hd, d),
                        "gu": w(d, 2 * m.d_ff), "down": w(m.d_ff, d)}
                       for _ in range(m.n_layers)]
        self.nf = torch.ones(d, dtype=dt, device=device)
        self.head = w(d, m.vocab)

    def caches(self, batch: int, context: int):
        torch = _torch()
        m = self.m
        shape = (batch, m.n_kv_heads, context, m.head_dim)
        return [(torch.randn(shape, dtype=self.dt, device=self.device) * 0.1,
                 torch.randn(shape, dtype=self.dt, device=self.device) * 0.1)
                for _ in range(m.n_layers)]

    def step(self, tokens, caches, pos: int):
        """One decode step writing KV at `pos` and attending over positions [0, pos]."""
        torch = _torch()
        F = torch.nn.functional
        m = self.m
        B, d, hd, h, hkv = tokens.shape[0], m.d_model, m.head_dim, m.n_heads, m.n_kv_heads
        g = h // hkv
        x = self.embed.index_select(0, tokens)
        for lw, (kc, vc) in zip(self.layers, caches, strict=True):
            y = F.rms_norm(x, (d,), lw["n1"])
            qkv = y @ lw["qkv"]
            q = qkv[:, : h * hd].view(B, hkv, g, hd)
            kc[:, :, pos] = qkv[:, h * hd: (h + hkv) * hd].view(B, hkv, hd)
            vc[:, :, pos] = qkv[:, (h + hkv) * hd:].view(B, hkv, hd)
            s = q @ kc[:, :, : pos + 1].transpose(-1, -2)  # (B, hkv, g, C); scale folded into W
            p = torch.softmax(s, dim=-1)
            a = (p @ vc[:, :, : pos + 1]).reshape(B, h * hd)
            x = x + a @ lw["o"]
            y = F.rms_norm(x, (d,), lw["n2"])
            gu = y @ lw["gu"]
            x = x + (F.silu(gu[:, : m.d_ff]) * gu[:, m.d_ff:]) @ lw["down"]
        return F.rms_norm(x, (d,), self.nf) @ self.head


def measure_twin(tw: Twin, device: str, seed: int = 0, log=print) -> list[dict]:
    """Median decode-step time per grid point, in seeded shuffled order, with the contention
    rule (re-measure once if CV > CV_LIMIT, then mark invalid)."""
    torch = _torch()
    model = TwinModel(tw, device, seed)
    pts = grid(tw)
    order = pts[:]
    random.Random(seed).shuffle(order)
    rows = {}
    for i, (b, c) in enumerate(order):
        caches = model.caches(b, c)
        tokens = torch.randint(0, tw.model.vocab, (b,), device=device)
        attempts = []
        for _ in range(2):
            ts = time_fn(lambda tokens=tokens, caches=caches, c=c: model.step(tokens, caches,
                                                                               c - 1), device)
            attempts.append(ts)
            if _cv(ts) <= CV_LIMIT:
                break
        ts = attempts[-1]
        rows[(b, c)] = {"batch": b, "context": c, "measured_s": statistics.median(ts),
                        "cv": _cv(ts), "valid": _cv(ts) <= CV_LIMIT, "attempts": len(attempts),
                        "trials_s": ts}
        del caches
        if device == "mps":
            torch.mps.empty_cache()
        log(f"  [{i + 1}/{len(order)}] batch {b:4d} context {c:6d}: "
            f"{rows[(b, c)]['measured_s'] * 1e3:8.3f} ms (cv {rows[(b, c)]['cv']:.2f})")
    return [rows[p] for p in pts]


def validate(tw: Twin, hw: HardwareSpec, measured: list[dict]) -> dict:
    control_hw = dataclasses.replace(hw, hbm_gbps=hw.hbm_gbps * 4)
    rows = []
    for r in measured:
        pred, share = predict(tw, hw, r["batch"], r["context"])
        rows.append({**r, "predicted_s": pred, "hbm_share": share,
                     "control_s": predict(tw, control_hw, r["batch"], r["context"])[0],
                     "structural_s": predict(tw, hw, r["batch"], 1)[0]})
    valid = [r for r in rows if r["valid"]]
    regimes = {}
    for name, sel in (("memory", lambda r: r["hbm_share"] >= 0.7),
                      ("other", lambda r: r["hbm_share"] < 0.7)):
        ape = [abs(r["predicted_s"] - r["measured_s"]) / r["measured_s"] for r in valid if sel(r)]
        regimes[name] = {"n": len(ape), "mdape": statistics.median(ape) if ape else None}
    struct = [abs(r["structural_s"] - r["measured_s"]) / r["measured_s"] for r in valid]
    return {"p1": grade(rows, len(rows)), "negative_control": grade_negative_control(rows),
            "structural_control_mdape": statistics.median(struct) if struct else None,
            "by_regime": regimes, "rows": rows}


# ------------------------------------------------------------------ CLI entry


def main(a) -> int:
    if a.measure_cmd == "calibrate":
        device = pick_device(a.device)
        if device is None:
            print(f"BLOCKED: device {a.device!r} is not available")
            return 3
        out = prepare_dir(Path(a.out), a.force)
        hw, ev = calibrate(device)
        write_files(out, {"profile.json": dumps(to_dict(hw)), "calibration.json": dumps(ev)},
                    {"kind": "calibration"})
        print(f"{hw.name}: copy {hw.hbm_gbps:.1f} GB/s, fp16 GEMM {hw.peak_tflops['fp16']:.2f} "
              f"TFLOP/s, launch {hw.kernel_overhead_us:.1f} us -> {out}/profile.json")
        return 0

    hw = load_hardware(a.profile)
    if a.set:
        hw = with_overrides(hw, parse_sets(a.set))
    tw = load_twin(a.twin)
    if not hw.name.startswith("local-"):
        raise SpecError(f"--profile: {hw.name} is not a calibrated local-<device> profile")
    device = hw.name.removeprefix("local-")
    if pick_device(device) is None:
        print(f"BLOCKED: profile device {device!r} is not available")
        return 3
    out = prepare_dir(Path(a.out), a.force)
    check = grid_check(tw, hw)
    print(f"twin {tw.model.name}: {check['points']} points, {check['hbm_share_ge_0.7']} predicted "
          f"HBM-bound (rule ok: {check['ok']})")
    measured = measure_twin(tw, device)
    res = validate(tw, hw, measured)
    res.update({"twin": tw.model.name, "twin_sha256": spec_hash(tw), "profile": to_dict(hw),
                "profile_sha256": spec_hash(hw), "grid_check": check,
                "device_name": _device_name(device), "torch": _torch().__version__,
                "platform": platform.platform()})
    write_files(out, {"validation.json": dumps(res)}, {"kind": "validation"})
    p, nc = res["p1"], res["negative_control"]
    if p["verdict"] != INSUFFICIENT:
        print(f"P1 {p['verdict']}: MdAPE {p['mdape']:.3f} (95% CI {p['mdape_ci95'][0]:.3f}-"
              f"{p['mdape_ci95'][1]:.3f}) over {p['valid_points']}/{p['points']} valid points; "
              f"knee measured {p['knee_measured']} predicted {p['knee_predicted']}")
    else:
        print(f"P1 {INSUFFICIENT}: {p['valid_points']}/{p['points']} valid points")
    if nc["mdape"] is not None:
        print(f"negative control (hbm x4): MdAPE {nc['mdape']:.3f} (CI low "
              f"{nc['mdape_ci95'][0]:.3f}) rejected={nc['rejected']}; structural control MdAPE "
              f"{res['structural_control_mdape']:.3f}")
    print(f"by regime: {json.dumps(res['by_regime'])}")
    return 0 if p["verdict"] == PASS and nc["rejected"] else 1
