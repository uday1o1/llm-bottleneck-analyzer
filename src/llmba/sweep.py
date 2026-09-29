"""Characterization sweeps: grid expansion, dataset rows, and a content-addressed manifest."""

from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import platform
from pathlib import Path

from . import __version__
from .analyze import capacity, request
from .cost import BOUNDS
from .specs import SpecError, Workload, load_hardware, load_model, spec_hash

COLUMNS = [
    "model", "hardware", "batch", "prompt_len", "output_len", "tp", "weight_dtype", "kv_dtype",
    "act_dtype", "fits", "max_batch", "weight_gb_per_rank", "kv_gb_per_rank", "ttft_s", "tpot_s",
    "e2e_s", "decode_tokens_per_s", "output_tokens_per_s", "prefill_bound", "decode_bound",
    *[f"elasticity_{b}" for b in BOUNDS],
]


def _fmt(v):
    if isinstance(v, bool):
        return str(v).lower()
    return format(v, ".9g") if isinstance(v, float) else v


def row(m, hw, wl) -> dict:
    wl.check(m)
    cap = capacity(m, hw, wl)
    r = request(m, hw, wl)
    el = r["elasticity"]
    dominant = {p: max(BOUNDS, key=lambda b, e=r[f"elasticity_{p}"]: (e[b], -BOUNDS.index(b)))
                for p in ("prefill", "decode")}
    out = {
        "model": m.name, "hardware": hw.name, **{k: getattr(wl, k) for k in (
            "batch", "prompt_len", "output_len", "tp", "weight_dtype", "kv_dtype", "act_dtype")},
        "fits": cap["fits"], "max_batch": cap["max_batch"],
        "weight_gb_per_rank": cap["weight_bytes"] / 1e9, "kv_gb_per_rank": cap["kv_bytes"] / 1e9,
        **{k: r[k] for k in ("ttft_s", "tpot_s", "e2e_s", "decode_tokens_per_s",
                             "output_tokens_per_s")},
        "prefill_bound": dominant["prefill"], "decode_bound": dominant["decode"],
        **{f"elasticity_{b}": el[b] for b in BOUNDS},
    }
    return out


def run_grid(grid: dict) -> tuple[list[dict], list[dict]]:
    """Expand a grid (lists per key) in deterministic order. Invalid combinations (tp that does
    not split heads, a dtype the hardware lacks) are skipped and reported, never silently lost."""
    keys = ["models", "hardware", "tp", "batch", "prompt_len", "output_len", "weight_dtype",
            "kv_dtype", "act_dtype"]
    defaults = {"tp": [1], "weight_dtype": ["bf16"], "kv_dtype": ["bf16"], "act_dtype": ["bf16"]}
    unknown = set(grid) - set(keys)
    if unknown:
        raise SpecError(f"grid.{sorted(unknown)[0]}: unknown key")
    axes = [grid.get(k, defaults.get(k)) for k in keys]
    if any(a is None for a in axes):
        missing = [k for k, a in zip(keys, axes, strict=True) if a is None]
        raise SpecError(f"grid.{missing[0]}: required")
    models = {n: load_model(n) for n in axes[0]}
    hws = {n: load_hardware(n) for n in axes[1]}
    rows, skipped = [], []
    for mn, hn, tp, b, s, o, wd, kd, ad in itertools.product(*axes):
        try:
            wl = Workload(batch=b, prompt_len=s, output_len=o, weight_dtype=wd, kv_dtype=kd,
                          act_dtype=ad, tp=tp)
            rows.append(row(models[mn], hws[hn], wl))
        except SpecError as e:
            skipped.append({"model": mn, "hardware": hn, "tp": tp, "batch": b, "prompt_len": s,
                            "output_len": o, "act_dtype": ad, "reason": str(e)})
    return rows, skipped


def to_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: _fmt(r[k]) for k in COLUMNS})
    return buf.getvalue()


def dumps(obj) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def prepare_dir(out: Path, force: bool = False) -> Path:
    """Create an output directory, refusing to clobber a non-empty one unless forced."""
    out = Path(out)
    if out.exists() and any(out.iterdir()) and not force:
        raise FileExistsError(f"{out} is not empty; choose a new directory or pass --force")
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_files(out: Path, files: dict[str, bytes | str], meta: dict) -> dict:
    """Write files plus manifest.json holding their SHA-256 and provenance (no timestamps, so
    identical inputs give identical bytes)."""
    digests = {}
    for name, data in sorted(files.items()):
        blob = data.encode() if isinstance(data, str) else data
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        (out / name).write_bytes(blob)
        digests[name] = hashlib.sha256(blob).hexdigest()
    manifest = {"llmba_version": __version__, "python": platform.python_version(),
                "files": digests, **meta}
    (out / "manifest.json").write_text(dumps(manifest))
    return manifest


def verify_manifest(out: Path) -> list[str]:
    """Names of files whose content no longer matches the manifest."""
    manifest = json.loads((Path(out) / "manifest.json").read_text())
    return [n for n, h in manifest["files"].items()
            if hashlib.sha256((Path(out) / n).read_bytes()).hexdigest() != h]


def sweep(grid: dict, out: Path, force: bool = False) -> dict:
    out = prepare_dir(out, force)
    rows, skipped = run_grid(grid)
    meta = {"grid": grid, "rows": len(rows), "skipped": skipped,
            "spec_hashes": {n: spec_hash(load_model(n)) for n in grid["models"]}
            | {n: spec_hash(load_hardware(n)) for n in grid["hardware"]}}
    return write_files(out, {"dataset.csv": to_csv(rows)}, meta)
