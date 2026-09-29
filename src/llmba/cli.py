"""Command-line interface. Exit codes: 0 ok, 1 gate FAIL/INCONCLUSIVE, 2 usage or spec error,
3 BLOCKED (requested device unavailable)."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from . import __version__
from .specs import (
    SpecError,
    Workload,
    load_hardware,
    load_model,
    parse_sets,
    preset_names,
    with_overrides,
)

STUDIES = Path(__file__).resolve().parents[2] / "studies"


def _csv(cast):
    return lambda s: [cast(x) for x in s.split(",") if x]


def _add_workload(p, batch=True):
    if batch:
        p.add_argument("--batch", type=int, default=1)
    p.add_argument("--prompt", type=int, default=1024, help="prompt tokens per sequence")
    p.add_argument("--output", type=int, default=256, help="generated tokens per sequence")
    p.add_argument("--tp", type=int, default=1, help="tensor-parallel degree")
    for k in ("weight", "kv", "act"):
        p.add_argument(f"--{k}-dtype", default="bf16", choices=["fp32", "bf16", "fp16", "fp8"])


def _add_point(p):
    p.add_argument("--model", required=True, help="preset name or JSON path")
    p.add_argument("--hw", required=True, help="preset name or JSON path")
    p.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                   help="override a hardware field, e.g. mem_eff=0.7 or peak_tflops.fp8=2000")
    _add_workload(p)
    p.add_argument("--json", action="store_true")


def _point(a):
    m = load_model(a.model)
    hw = load_hardware(a.hw)
    if a.set:
        hw = with_overrides(hw, parse_sets(a.set))
    wl = Workload(batch=a.batch, prompt_len=a.prompt, output_len=a.output, tp=a.tp,
                  weight_dtype=a.weight_dtype, kv_dtype=a.kv_dtype, act_dtype=a.act_dtype)
    wl.check(m)
    return m, hw, wl


def _print_json(obj):
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def _si(x, unit):
    for f, p in ((1e-6, "u"), (1e-3, "m"), (1, ""), (1e3, "k"), (1e6, "M"), (1e9, "G")):
        if abs(x) < f * 1000:
            return f"{x / f:.3g} {p}{unit}"
    return f"{x / 1e12:.3g} T{unit}"


def cmd_presets(a):
    for kind in ("models", "hardware", "twins"):
        print(f"{kind}: {', '.join(preset_names(kind))}")


def cmd_analyze(a):
    from .analyze import analyze

    m, hw, wl = _point(a)
    r = analyze(m, hw, wl)
    if a.json:
        return _print_json(r)
    cap, mt = r["capacity"], r["metrics"]
    print(f"{m.name} on {hw.name}  batch={wl.batch} prompt={wl.prompt_len} "
          f"output={wl.output_len} tp={wl.tp} weights={wl.weight_dtype} kv={wl.kv_dtype}")
    if hw.unverified:
        print(f"  note: unverified hardware fields: {', '.join(hw.unverified)}")
    print(f"  params {r['params']['total'] / 1e9:.2f}B total, {r['params']['active'] / 1e9:.2f}B "
          f"active; weights/rank {cap['weight_bytes'] / 1e9:.1f} GB; KV {cap['kv_bytes'] / 1e9:.1f}"
          f" GB; fits={cap['fits']} (max batch {cap['max_batch']})")
    print(f"  TTFT {_si(mt['ttft_s'], 's')}  TPOT {_si(mt['tpot_s'], 's')}  "
          f"e2e {_si(mt['e2e_s'], 's')}  decode {mt['decode_tokens_per_s']:.0f} tok/s  "
          f"output {mt['output_tokens_per_s']:.0f} tok/s")
    for phase in ("prefill", "decode"):
        el = r["elasticity"][phase]
        shares = "  ".join(f"{k} {v:.0%}" for k, v in el.items() if v > 0.005)
        print(f"\n  {phase}: time share (= elasticity) {shares}")
        print(f"    {'op':16s} {'share':>6s} {'bound':>9s}")
        for row in r["critical_path"][phase]:
            print(f"    {row['op']:16s} {row['share']:6.1%} {row['bound']:>9s}")
    print(f"\n  ridge intensity {r['ridge_intensity']:.0f} FLOP/byte")


def cmd_sweep(a):
    from .sweep import sweep

    if a.grid:
        grid = json.loads(Path(a.grid).read_text())
    else:
        grid = {"models": a.models, "hardware": a.hw, "batch": a.batch, "prompt_len": a.prompt,
                "output_len": a.output, "tp": a.tp, "weight_dtype": [a.weight_dtype],
                "kv_dtype": [a.kv_dtype], "act_dtype": [a.act_dtype]}
    man = sweep(grid, Path(a.out), a.force)
    print(f"wrote {man['rows']} rows ({len(man['skipped'])} skipped) to {a.out}/dataset.csv")
    print(f"sha256 {man['files']['dataset.csv']}")


def cmd_intercepts(a):
    from .intercepts import (
        attention_context,
        capacity_wall,
        decode_knee,
        tier2_breakeven,
        tp_saturation,
    )

    m, hw, wl = _point(a)
    r = {"model": m.name, "hardware": hw.name, "workload": dataclasses.asdict(wl),
         "decode_knee": decode_knee(m, hw, wl),
         "attention_context": attention_context(m, hw, wl),
         "capacity_wall": capacity_wall(m, hw, wl),
         "tp_saturation": tp_saturation(m, hw, wl) if hw.link_gbps else None,
         "tier2_breakeven": tier2_breakeven(m, hw, wl)}
    if a.json:
        return _print_json(r)
    k = r["decode_knee"]
    print(f"{m.name} on {hw.name} (context {k['context']}, tp {wl.tp}, weights {wl.weight_dtype})")
    print(f"  decode knee (compute >= 50% of step): {k['value'] or k['status']} "
          f"in batch range {k['range']}")
    print(f"  weight-GEMM knee: {k['gemm_knee'] or 'INCONCLUSIVE'}; analytic GEMM ridge "
          f"{k['analytic_gemm_ridge']:.0f}")
    ac = r["attention_context"]
    print(f"  attention >= 50% of decode at batch {wl.batch}: context "
          f"{ac['value'] or ac['status']}")
    print("  capacity wall: " + ", ".join(f"{x['context']}->{x['max_batch']}"
                                           for x in r["capacity_wall"]))
    if r["tp_saturation"]:
        ts = r["tp_saturation"]
        print(f"  TP saturation: {ts['value'] or ts['status']} "
              f"(TPOT {', '.join(f'tp{k}={_si(v, chr(115))}' for k, v in ts['tpot_s'].items())})")
    tb = r["tier2_breakeven"]
    print(f"  capacity tier ({tb['tier2_gb']:.0f} GB) helps above: "
          f"{str(tb['value']) + ' GB/s' if tb['value'] else tb['status']}")


def cmd_recommend(a):
    from .intercepts import recommend, study_cases

    if a.study:
        study = json.loads(Path(a.study).read_text())
        out = [recommend(m, hw, wl, name) for name, m, hw, wl in study_cases(study)]
    elif a.model and a.hw:
        m, hw, wl = _point(a)
        out = [recommend(m, hw, wl)]
    else:
        raise SpecError("recommend: pass --study, or both --model and --hw")
    if a.json:
        return _print_json(out)
    for r in out:
        wl = r["workload"]
        print(f"[{r['archetype'] or 'point'}] {r['model']} on {r['hardware']} "
              f"(prompt {wl['prompt_len']}, output {wl['output_len']}, batch {wl['batch']:g}, "
              f"tp {wl['tp']})")
        print(f"  {r['summary']}")
        for w in r["what_ifs"]:
            lat = f"x{w['latency_speedup']:.2f}" if w["latency_speedup"] else "n/a"
            tp_ = f"x{w['throughput_gain']:.2f}" if w["throughput_gain"] else "n/a"
            print(f"    {w['what_if']:34s} throughput {tp_:>6s}  latency {lat:>6s}")


def cmd_report(a):
    from .report import build_report

    out = build_report(json.loads(Path(a.study).read_text()), Path(a.out), a.force)
    print(f"wrote {out}/report.md")


def cmd_check_published(a):
    from .analyze import check_published

    r = check_published(json.loads(Path(a.reference).read_text()), load_model, load_hardware)
    if a.json:
        _print_json(r)
    else:
        print(f"S3 published-ceiling check vs {r['source_url']}")
        for x in r["rows"]:
            print(f"  {x['model']:14s} {x['hardware']:9s} x{x['n_gpus']} {x['prompt_len']:>6d}/"
                  f"{x['output_len']:<5d} measured {x['output_tokens_per_s']:9.1f}  ceiling "
                  f"{x['ceiling_tokens_per_s']:9.1f}  ratio {x['ratio']:.3f} "
                  f"({x['ceiling_bound']})")
        print(f"  {r['n']} rows, ratio min {r['min_ratio']:.2f} median {r['median_ratio']:.2f} "
              f"max {r['max_ratio']:.2f}: {r['verdict']}")
    return 0 if r["verdict"] == "PASS" else 1


def cmd_measure(a):
    from . import measure

    return measure.main(a)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="llmba", description=__doc__.splitlines()[0])
    ap.add_argument("--version", action="version", version=f"llmba {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("presets", help="list bundled presets").set_defaults(fn=cmd_presets)

    p = sub.add_parser("analyze", help="characterize one operating point")
    _add_point(p)
    p.set_defaults(fn=cmd_analyze)

    p = sub.add_parser("sweep", help="synthesize a characterization dataset over a grid")
    p.add_argument("--grid", help="grid JSON (overrides the list options)")
    p.add_argument("--models", type=_csv(str), default=["llama-3.1-8b"])
    p.add_argument("--hw", type=_csv(str), default=["h100-sxm"])
    p.add_argument("--batch", type=_csv(int), default=[1, 8, 64, 256])
    p.add_argument("--prompt", type=_csv(int), default=[1024])
    p.add_argument("--output", type=_csv(int), default=[256])
    p.add_argument("--tp", type=_csv(int), default=[1])
    for k in ("weight", "kv", "act"):
        p.add_argument(f"--{k}-dtype", default="bf16")
    p.add_argument("--out", required=True)
    p.add_argument("--force", action="store_true", help="allow a non-empty output directory")
    p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("intercepts", help="crossover points for one model and hardware")
    _add_point(p)
    p.set_defaults(fn=cmd_intercepts)

    p = sub.add_parser("recommend", help="ranked what-if intercepts")
    p.add_argument("--study", help="study JSON with 'cases'")
    p.add_argument("--model")
    p.add_argument("--hw")
    p.add_argument("--set", action="append", default=[])
    _add_workload(p)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_recommend)

    p = sub.add_parser("report", help="full Markdown report with figures")
    p.add_argument("--study", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("check-published", help="claim S3: published throughput vs ceiling")
    p.add_argument("--reference", default=str(STUDIES / "published_reference.json"))
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_check_published)

    p = sub.add_parser("measure", help="calibrate and validate on local hardware (torch)")
    msub = p.add_subparsers(dest="measure_cmd", required=True)
    c = msub.add_parser("calibrate", help="microbenchmark a device into a hardware profile")
    c.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    c.add_argument("--out", required=True)
    c.add_argument("--force", action="store_true")
    v = msub.add_parser("validate", help="predict and measure an executable twin")
    v.add_argument("--profile", required=True, help="profile.json from calibrate")
    v.add_argument("--twin", required=True, help="twin preset name or JSON path")
    v.add_argument("--out", required=True)
    v.add_argument("--force", action="store_true")
    v.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                   help="override a profile field before predicting (e.g. overhead_mode=overlap)")
    for q in (c, v):
        q.set_defaults(fn=cmd_measure)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return a.fn(a) or 0
    except (SpecError, FileExistsError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
