"""Study report: results.json (all numbers), Markdown rendered from it, and figures."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from . import __version__
from .analyze import capacity, check_published, request, step_costs
from .cost import BOUNDS
from .intercepts import (
    decode_knee,
    max_throughput,
    recommend,
    study_cases,
    tier2_breakeven,
)
from .ops import DECODE, PREFILL
from .specs import Workload, load_hardware, load_model, spec_hash
from .sweep import dumps, prepare_dir, run_grid, to_csv, write_files

STUDIES = Path(__file__).resolve().parents[2] / "studies"

# Validated categorical order (blue, orange, aqua, yellow), recessive ink and grid.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


# ------------------------------------------------------------------ computation


def _wl(spec: dict, **kw) -> Workload:
    keys = {f.name for f in dataclasses.fields(Workload)}
    return Workload(**{k: v for k, v in {**spec, **kw}.items() if k in keys})


def compute(study: dict) -> dict:
    res: dict = {"llmba_version": __version__, "title": study["title"]}

    # Roofline: ops of one model at three operating points.
    f0 = study["focus"][0]
    m, hw = load_model(f0["model"]), load_hardware(f0["hardware"])
    pts = []
    for label, phase, b in (("decode, batch 1", DECODE, 1), ("decode, batch 256", DECODE, 256),
                            ("prefill, batch 1", PREFILL, 1)):
        wl = _wl(f0, batch=b, prompt_len=1024)
        for c in step_costs(m, hw, wl, phase, 1024):
            if c.op.flops and c.op.hbm_bytes:
                pts.append({"series": label, "op": c.op.name,
                            "intensity": c.op.flops / c.op.hbm_bytes,
                            "tflops": c.op.flops / (c.t_busy + c.t_overhead) / 1e12,
                            "bound": c.bound})
    res["roofline"] = {"model": m.name, "hardware": hw.name, "peak_tflops": hw.peak("bf16") / 1e12,
                       "eff_peak_tflops": hw.peak("bf16") * hw.compute_eff / 1e12,
                       "eff_bw_tbps": hw.hbm_gbps * hw.mem_eff / 1e3, "points": pts}

    # Decode scaling across hardware.
    sc = study["scaling"]
    m = load_model(sc["model"])
    scaling = []
    for hn in sc["hardware"]:
        hw = load_hardware(hn)
        wl = Workload(prompt_len=sc["context"], output_len=1)
        k = decode_knee(m, hw, wl)
        curve = []
        for b in sc["batch"]:
            w = dataclasses.replace(wl, batch=b)
            if not capacity(m, hw, w)["fits"]:
                break
            costs = step_costs(m, hw, w, DECODE, sc["context"])
            curve.append({"batch": b, "tokens_per_s": b / sum(c.time for c in costs)})
        scaling.append({"hardware": hn, "gemm_knee": k["gemm_knee"], "step_knee": k["value"],
                        "max_batch": k["range"][1], "curve": curve,
                        "analytic_gemm_ridge": k["analytic_gemm_ridge"]})
    res["scaling"] = {"model": m.name, "context": sc["context"], "rows": scaling}

    # Intercept map: capacity wall versus weight-GEMM knee over context.
    im = study["intercept_map"]
    hw = load_hardware(im["hardware"])
    imap = []
    for spec in im["models"]:
        m = load_model(spec["model"])
        wall = []
        for c in im["context"]:
            w = _wl(spec, prompt_len=c, output_len=1)
            wall.append({"context": c, "max_batch": capacity(m, hw, w)["max_batch"]})
        knee = decode_knee(m, hw, _wl(spec, prompt_len=im["context"][0]))
        imap.append({**spec, "gemm_knee": knee["gemm_knee"],
                     "unconstrained_gemm_knee": knee["unconstrained_gemm_knee"],
                     "analytic_gemm_ridge": knee["analytic_gemm_ridge"], "wall": wall,
                     "knee_reachable_up_to_context": max(
                         (x["context"] for x in wall
                          if knee["gemm_knee"] and x["max_batch"] >= knee["gemm_knee"]),
                         default=None)})
    res["intercept_map"] = {"hardware": hw.name, "models": imap}

    # Archetype recommendations.
    recs = []
    for name, m, hw, wl in study_cases(study):
        r = recommend(m, hw, wl, name)
        lat = request(m, hw, wl) if capacity(m, hw, wl)["fits"] else None
        recs.append({**r, "ttft_s": lat and lat["ttft_s"], "tpot_s": lat and lat["tpot_s"],
                     "max_tokens_per_s": max_throughput(m, hw, wl)})
    res["recommendations"] = recs

    # Capacity-tier intercept for the first focus pair.
    t2 = study["tier2"]
    tier = []
    for name, m, hw, wl in study_cases(study)[: len(study["archetypes"])]:
        base = max_throughput(m, hw, wl)
        gains = [{"gbps": g, "gain": max_throughput(m, dataclasses.replace(
            hw, tier2_gb=float(t2["gb"]), tier2_gbps=float(g)), wl) / base} for g in t2["gbps"]]
        be = tier2_breakeven(m, hw, wl, float(t2["gb"]))
        tier.append({"archetype": name, "model": m.name, "hardware": hw.name,
                     "breakeven_gbps": be["value"], "status": be["status"], "gains": gains})
    res["tier2"] = {"tier2_gb": t2["gb"], "rows": tier}

    # Published ceiling (claim S3).
    ref = json.loads((STUDIES / "published_reference.json").read_text())
    res["published"] = check_published(ref, load_model, load_hardware)
    res["published"]["note"] = (
        "Ceiling per request: max(FLOPs / fp8 peak, KV_req * (1 + W_read / (Cap - W)) / HBM "
        "bandwidth), with binary-GiB capacity, 1-byte weights and KV, efficiency 1 and zero "
        "overhead; it holds for any batching schedule.")

    res["spec_hashes"] = {n: spec_hash(load_model(n)) for n in study["sweep"]["models"]} | {
        n: spec_hash(load_hardware(n)) for n in study["sweep"]["hardware"]}
    return res


# ------------------------------------------------------------------ figures


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_facecolor(SURFACE)


def _save(fig, path: Path):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=150, metadata={"Software": None})


def figures(res: dict, out: Path) -> list[str]:
    try:
        import matplotlib
    except ImportError as e:
        raise RuntimeError("figures need matplotlib: pip install '.[plot]'") from e
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = []

    # 1. Roofline.
    rf = res["roofline"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    xs = [0.1, 1e4]
    ridge = rf["eff_peak_tflops"] / rf["eff_bw_tbps"]
    ax.plot([xs[0], ridge, xs[1]], [xs[0] * rf["eff_bw_tbps"], rf["eff_peak_tflops"],
                                     rf["eff_peak_tflops"]], color=INK, linewidth=1.5)
    ax.annotate(f"ridge {ridge:.0f} FLOP/B", (ridge, rf["eff_peak_tflops"]),
                textcoords="offset points", xytext=(-8, 8), ha="right", fontsize=8, color=INK2)
    for i, series in enumerate(dict.fromkeys(p["series"] for p in rf["points"])):
        p = [x for x in rf["points"] if x["series"] == series]
        ax.scatter([x["intensity"] for x in p], [x["tflops"] for x in p], s=40, color=SERIES[i],
                   edgecolors=SURFACE, linewidths=1.5, label=series, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style(ax, f"Roofline: {rf['model']} ops on {rf['hardware']}\n(bf16, default efficiency "
               "knobs)",
           "arithmetic intensity (FLOP / HBM byte)", "attained TFLOP/s")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out / "roofline.png")
    plt.close(fig)
    names.append("roofline.png")

    # 2. Decode throughput versus batch across hardware.
    sc = res["scaling"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, row in enumerate(sc["rows"]):
        b = [x["batch"] for x in row["curve"]]
        t = [x["tokens_per_s"] for x in row["curve"]]
        ax.plot(b, t, color=SERIES[i], linewidth=2, marker="o", markersize=4, label=row["hardware"])
        ax.annotate(row["hardware"], (b[-1], t[-1]), textcoords="offset points", xytext=(5, 0),
                    fontsize=8, color=INK2, va="center")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    _style(ax, f"Decode throughput vs batch: {sc['model']}, context {sc['context']}\n"
               "(each curve ends at its capacity limit)", "batch", "decode tokens/s")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out / "decode_scaling.png")
    plt.close(fig)
    names.append("decode_scaling.png")

    # 3. Intercept map: capacity wall versus weight-GEMM knee.
    im = res["intercept_map"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, row in enumerate(im["models"]):
        ctx = [x["context"] for x in row["wall"] if x["max_batch"] > 0]
        mb = [x["max_batch"] for x in row["wall"] if x["max_batch"] > 0]
        name = f"{row['model']} (tp {row.get('tp', 1)})"
        ax.plot(ctx, mb, color=SERIES[i], linewidth=2, marker="o", markersize=4,
                label=f"{name} capacity wall")
        knee = row["unconstrained_gemm_knee"]
        ax.axhline(knee, color=SERIES[i], linewidth=1.2, linestyle=(0, (4, 3)))
        ax.annotate(f"{name} GEMM knee {knee}", (im["models"][0]["wall"][-1]["context"], knee),
                    textcoords="offset points", xytext=(0, 3), ha="right", fontsize=7,
                    color=INK2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    _style(ax, f"Intercept map on {im['hardware']}\ndecode GEMMs are compute-bound only "
               "where the capacity wall is above the knee", "context length (tokens)", "batch")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    _save(fig, out / "intercept_map.png")
    plt.close(fig)
    names.append("intercept_map.png")

    # 4. Throughput elasticity heatmap (archetype x resource), first focus pair.
    recs = [r for r in res["recommendations"] if r["model"] == res["recommendations"][0]["model"]]
    keys = ["compute", "hbm", "capacity", "network", "overhead"]
    grid = [[r["throughput_elasticity"].get(k, 0.0) for k in keys] for r in recs]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for y, row in enumerate(grid):
        for x, v in enumerate(row):
            ax.text(x, y, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if v > 0.5 else INK)
    ax.set_xticks(range(len(keys)), keys)
    ax.set_yticks(range(len(recs)), [r["archetype"] for r in recs])
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Throughput elasticity at the capacity-limited batch\n{recs[0]['model']} on "
                 f"{recs[0]['hardware']}", loc="left", fontsize=10, color=INK, pad=10)
    _save(fig, out / "elasticity.png")
    plt.close(fig)
    names.append("elasticity.png")

    # 5. Capacity-tier gain versus tier bandwidth.
    t2 = res["tier2"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    shown = [r for r in t2["rows"] if max(g["gain"] for g in r["gains"]) > 1.001][:4]
    for i, row in enumerate(shown):
        g = [x["gbps"] for x in row["gains"]]
        v = [x["gain"] for x in row["gains"]]
        ax.plot(g, v, color=SERIES[i], linewidth=2, marker="o", markersize=4,
                label=row["archetype"])
        ax.annotate(row["archetype"], (g[-1], v[-1]), textcoords="offset points",
                    xytext=(5, 0), fontsize=8, color=INK2, va="center")
    ax.axhline(1.0, color=INK2, linewidth=1)
    ax.set_xscale("log", base=2)
    first = t2["rows"][0]
    _style(ax, f"{t2['tier2_gb']} GB capacity tier: throughput gain vs tier bandwidth\n"
               f"{first['model']} on {first['hardware']} (first four archetypes with a gain)",
           "tier-2 bandwidth (GB/s)",
           "throughput vs HBM only (x)")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out / "tier2.png")
    plt.close(fig)
    names.append("tier2.png")

    # 6. Published measured throughput as a fraction of the ceiling.
    pub = res["published"]["rows"]
    groups = list(dict.fromkeys((r["model"], r["hardware"]) for r in pub))
    cases = list(dict.fromkeys(f"{r['prompt_len']}/{r['output_len']}" for r in pub))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, gkey in enumerate(groups):
        rows = [r for r in pub if (r["model"], r["hardware"]) == gkey]
        ax.scatter([r["ratio"] for r in rows],
                   [cases.index(f"{r['prompt_len']}/{r['output_len']}") + (i - 1.5) * 0.12
                    for r in rows], s=40, color=SERIES[i], edgecolors=SURFACE, linewidths=1.5,
                   label=f"{gkey[0]} / {gkey[1]}", zorder=3)
    ax.axvline(1.0, color=INK, linewidth=1.2)
    ax.annotate("ceiling", (1.0, len(cases) - 0.6), xytext=(-40, 0), textcoords="offset points",
                fontsize=8, color=INK2)
    ax.set_xlim(0, 1.3)
    ax.set_yticks(range(len(cases)), [f"ISL/OSL {c}" for c in cases])
    _style(ax, "Published TensorRT-LLM FP8 throughput\nas a fraction of the scheduler-independent "
               "ceiling",
           "measured / ceiling", "")
    ax.legend(frameon=False, fontsize=7, loc="center right")
    _save(fig, out / "published.png")
    plt.close(fig)
    names.append("published.png")
    return names


# ------------------------------------------------------------------ markdown


def _x(v, fmt="{:.2f}", none="n/a"):
    return none if v is None else fmt.format(v)


def render(res: dict, sweep_meta: dict) -> str:
    L = [f"# {res['title']}", "",
         f"Generated by llmba {res['llmba_version']}. Every number below is read from "
         "`results.json`; datacenter GPU numbers are analytical model outputs from datasheet "
         "specs, not measurements.", ""]

    L += ["## Findings", ""]
    im = res["intercept_map"]
    for row in im["models"]:
        if row["gemm_knee"] and row["knee_reachable_up_to_context"]:
            L.append(f"- {row['model']} on {im['hardware']}: weight GEMMs turn compute-bound at "
                     f"batch {row['gemm_knee']}, which fits in memory only up to context "
                     f"{row['knee_reachable_up_to_context']}.")
        else:
            L.append(f"- {row['model']} on {im['hardware']}: weight GEMMs would turn compute-bound "
                     f"only at batch {row['unconstrained_gemm_knee']}, above the capacity wall at "
                     "every studied context, so decode stays memory-bound whatever the batch; "
                     "bandwidth and capacity, not FLOPs, set its decode throughput.")
    for r in res["recommendations"]:
        if r["archetype"] in ("reasoning-long-decode", "rag-long-context"):
            L.append(f"- {r['archetype']} ({r['model']} on {r['hardware']}): {r['summary']}")
    for row in res["tier2"]["rows"]:
        if row["breakeven_gbps"]:
            L.append(f"- A {res['tier2']['tier2_gb']} GB capacity tier raises {row['archetype']} "
                     f"throughput on {row['hardware']} only above {row['breakeven_gbps']} GB/s.")
    p = res["published"]
    L += [f"- Claim S3 {p['verdict']}: {p['n']} published FP8 throughput rows reach "
          f"{p['min_ratio']:.2f} to {p['max_ratio']:.2f} of the scheduler-independent ceiling "
          f"(median {p['median_ratio']:.2f}).", ""]

    rf = res["roofline"]
    L += ["## Roofline", "", "![roofline](roofline.png)", "",
          f"{rf['model']} ops on {rf['hardware']}; calibrated-efficiency roof "
          f"{rf['eff_peak_tflops']:.0f} TFLOP/s and {rf['eff_bw_tbps']:.2f} TB/s.", ""]

    sc = res["scaling"]
    L += ["## Decode scaling across hardware", "", "![decode scaling](decode_scaling.png)", "",
          "| hardware | weight-GEMM knee | whole-step knee | max batch (fits) | analytic ridge |",
          "| --- | ---: | ---: | ---: | ---: |"]
    for r in sc["rows"]:
        L.append(f"| {r['hardware']} | {_x(r['gemm_knee'], '{}', 'none')} | "
                 f"{_x(r['step_knee'], '{}', 'none')} | {r['max_batch']} | "
                 f"{r['analytic_gemm_ridge']:.0f} |")
    L += ["", "Whole-step knee `none` means attention KV reads keep the step memory-bound at "
          "every batch that fits (GQA attention intensity does not grow with batch).", ""]

    L += ["## Intercept map", "", "![intercept map](intercept_map.png)", "",
          "| model | tp | weight-GEMM knee (ignoring capacity) | knee reachable up to context |",
          "| --- | ---: | ---: | ---: |"]
    for row in im["models"]:
        L.append(f"| {row['model']} | {row.get('tp', 1)} | {row['unconstrained_gemm_knee']} | "
                 f"{_x(row['knee_reachable_up_to_context'], '{}', 'never')} |")
    L += ["", "MoE layers stream the weights of every expert the batch touches, so their knee "
          "sits roughly `n_experts / top_k` times above a dense layer's until all experts are "
          "active.", ""]

    L += ["## Archetype recommendations", "",
          "Latency is end-to-end at the archetype batch; throughput is output tokens/s at the "
          "best batch that fits (static batching). What-if gains are throughput multipliers.", "",
          "| archetype | model | hardware | TTFT | TPOT | max tok/s | top what-if | gain | "
          "most sensitive resource |",
          "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- |"]
    for r in res["recommendations"]:
        top = r["what_ifs"][0]
        tel = r["throughput_elasticity"]
        sens = max(tel, key=tel.get) if tel else "n/a"
        L.append(f"| {r['archetype']} | {r['model']} | {r['hardware']} | "
                 f"{_x(r['ttft_s'], '{:.3g} s')} | "
                 f"{_x(r['tpot_s'] and r['tpot_s'] * 1e3, '{:.3g} ms')}"
                 f" | {r['max_tokens_per_s']:.0f} | {top['what_if']} | "
                 f"{_x(top['throughput_gain'], 'x{:.2f}')} | {sens} ({_x(tel.get(sens))}) |")
    L += ["", "![elasticity](elasticity.png)", "",
          "A capacity elasticity above 1 means weights occupy most of HBM, so each percent of "
          "extra capacity adds more than a percent of KV room and batch.", ""]

    t2 = res["tier2"]
    L += ["## Capacity-tier intercept", "", "![tier 2](tier2.png)", "",
          f"| archetype | break-even tier bandwidth for a {t2['tier2_gb']} GB tier |",
          "| --- | ---: |"]
    for row in t2["rows"]:
        L.append(f"| {row['archetype']} | "
                 f"{_x(row['breakeven_gbps'], '{} GB/s', 'no gain at any bandwidth')} |")
    L.append("")

    L += ["## Published ceiling check (claim S3)", "", "![published](published.png)", "",
          f"Source: {p['source_url']}. {p['note']}", "",
          "| model | hardware | GPUs | ISL/OSL | measured tok/s | ceiling tok/s | ratio | "
          "ceiling bound |", "| --- | --- | ---: | --- | ---: | ---: | ---: | --- |"]
    for x in p["rows"]:
        L.append(f"| {x['model']} | {x['hardware']} | {x['n_gpus']} | {x['prompt_len']}/"
                 f"{x['output_len']} | {x['output_tokens_per_s']:.1f} | "
                 f"{x['ceiling_tokens_per_s']:.1f} | {x['ratio']:.3f} | {x['ceiling_bound']} |")
    L.append("")

    L += ["## Synthesized dataset", "",
          f"`dataset.csv`: {sweep_meta['rows']} operating points "
          f"({len(sweep_meta['skipped'])} invalid combinations skipped and listed in "
          "`manifest.json`). Columns include phase times, capacity, bound classes, and "
          f"elasticities for {', '.join(BOUNDS)}.", ""]

    L += ["## Assumptions and limitations", "",
          "- First-order analytical model: compulsory HBM traffic, no on-chip cache model, no "
          "cycle accuracy.",
          "- Efficiency knobs (compute 0.7, memory 0.8, network 0.8, 4 us per kernel) are "
          "assumed defaults, not datasheet values.",
          "- MoE assumes uniform independent routing when counting distinct experts.",
          "- Throughput assumes static batching; chunked prefill and in-flight batching can do "
          "better (the S3 ceiling accounts for that, the what-if objective does not).",
          "- Hardware fields marked unverified in presets: "
          + ", ".join(f"{n}: {', '.join(load_hardware(n).unverified)}"
                      for n in sorted(res["spec_hashes"]) if n in _hw_names()
                      and load_hardware(n).unverified) + ".", ""]
    return "\n".join(L)


def _hw_names():
    from .specs import preset_names
    return set(preset_names("hardware"))


def build_report(study: dict, out: Path, force: bool = False) -> Path:
    out = prepare_dir(out, force)
    res = compute(study)
    rows, skipped = run_grid(study["sweep"])
    sweep_meta = {"rows": len(rows), "skipped": skipped, "grid": study["sweep"]}
    files = {"results.json": dumps(res), "dataset.csv": to_csv(rows),
             "report.md": render(res, sweep_meta)}
    for name in figures(res, out):
        files[name] = (out / name).read_bytes()
    write_files(out, files, {"study": study, **sweep_meta})
    return out


def plot_validation(validation: dict, out: Path) -> Path:
    """Predicted versus measured decode-step time from a `measure validate` result (plot only)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in validation["rows"] if r["valid"]]
    fig, ax = plt.subplots(figsize=(6, 5))
    lo = min(min(r["measured_s"], r["predicted_s"]) for r in rows) * 1e3 * 0.8
    hi = max(max(r["measured_s"], r["predicted_s"]) for r in rows) * 1e3 * 1.2
    ax.plot([lo, hi], [lo, hi], color=INK, linewidth=1.2)
    ax.fill_between([lo, hi], [lo * 0.7, hi * 0.7], [lo * 1.3, hi * 1.3], color=GRID, alpha=0.6,
                    linewidth=0)
    for i, (name, sel) in enumerate((("predicted HBM-bound", lambda r: r["hbm_share"] >= 0.7),
                                     ("other", lambda r: r["hbm_share"] < 0.7))):
        pts = [r for r in rows if sel(r)]
        ax.scatter([r["measured_s"] * 1e3 for r in pts], [r["predicted_s"] * 1e3 for r in pts],
                   s=40, color=SERIES[i], edgecolors=SURFACE, linewidths=1.5, label=name,
                   zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    p = validation["p1"]
    _style(ax, f"{validation['twin']} on {validation['device_name']}: MdAPE {p['mdape']:.3f} "
               f"({p['verdict']})\nshaded band: +/-30 percent", "measured step (ms)",
           "predicted step (ms)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _save(fig, out)
    plt.close(fig)
    return out
