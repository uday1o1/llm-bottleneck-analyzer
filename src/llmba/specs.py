"""Model, hardware, and workload specs: strict JSON loading, validation, and presets."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_VERSION = 1
PRESETS = Path(__file__).parent / "presets"
DTYPE_BYTES = {"fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1}
MAX_SPEC_BYTES = 1_000_000


class SpecError(ValueError):
    """Invalid spec; the message names the offending field."""


@dataclass(frozen=True)
class MoE:
    n_experts: int
    top_k: int
    d_ff_expert: int
    n_shared: int = 0
    first_dense: int = 0  # leading layers that use the dense MLP of width d_ff


@dataclass(frozen=True)
class MLA:
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int


@dataclass(frozen=True)
class ModelSpec:
    name: str
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    vocab: int
    tied_embeddings: bool = False
    moe: MoE | None = None
    mla: MLA | None = None
    published_params_b: float | None = None
    published_active_params_b: float | None = None
    source_urls: tuple[str, ...] = ()
    notes: str = ""
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        for f in ("n_layers", "d_model", "n_heads", "n_kv_heads", "head_dim", "d_ff", "vocab"):
            _positive(self, f)
        if self.n_heads % self.n_kv_heads:
            raise SpecError(f"n_kv_heads: {self.n_heads} heads not divisible by {self.n_kv_heads}")
        if self.moe:
            for f in ("n_experts", "top_k", "d_ff_expert"):
                _positive(self.moe, f, "moe.")
            if not self.moe.top_k <= self.moe.n_experts:
                raise SpecError("moe.top_k: must not exceed moe.n_experts")
            if not 0 <= self.moe.first_dense <= self.n_layers:
                raise SpecError("moe.first_dense: must be within [0, n_layers]")
        if self.mla:
            for f in dataclasses.fields(self.mla):
                _positive(self.mla, f.name, "mla.")


@dataclass(frozen=True)
class HardwareSpec:
    name: str
    peak_tflops: dict[str, float]  # dense tensor throughput per dtype
    hbm_gb: float  # decimal GB, datasheet value
    hbm_gbps: float  # 1e9 bytes/s
    link_gbps: float = 0.0  # scale-up interconnect, per GPU, per direction
    memory_type: str = ""
    # Optional capacity tier (e.g. LPDDR over chip-to-chip link, CXL) holding KV spilled past HBM.
    tier2_gb: float = 0.0
    tier2_gbps: float = 0.0
    # Calibration knobs: assumed defaults, not datasheet values. Override per target.
    compute_eff: float = 0.7
    mem_eff: float = 0.8
    net_eff: float = 0.8
    kernel_overhead_us: float = 4.0
    link_latency_us: float = 5.0
    capacity_usable: float = 0.9
    overhead_mode: str = "additive"  # additive | overlap (dispatch hidden behind busy kernels)
    # Attained fraction of roofline for matmuls by row count M (calibrated; empty means 1.0).
    gemm_eff_by_tokens: dict[str, float] = field(default_factory=dict)
    attn_eff: float = 1.0  # attained fraction of roofline for decode attention (calibrated)
    unverified: tuple[str, ...] = ()
    source_urls: tuple[str, ...] = ()
    notes: str = ""
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        if not self.peak_tflops:
            raise SpecError("peak_tflops: at least one dtype required")
        for dt, v in self.peak_tflops.items():
            if dt not in DTYPE_BYTES:
                raise SpecError(f"peak_tflops.{dt}: unknown dtype")
            if not v > 0:
                raise SpecError(f"peak_tflops.{dt}: must be > 0")
        for f in ("hbm_gb", "hbm_gbps"):
            _positive(self, f)
        if self.overhead_mode not in ("additive", "overlap"):
            raise SpecError("overhead_mode: must be 'additive' or 'overlap'")
        if (self.tier2_gb > 0) != (self.tier2_gbps > 0):
            raise SpecError("tier2_gbps: tier2_gb and tier2_gbps must both be set or both be 0")
        if not 0 < self.attn_eff <= 2:
            raise SpecError("attn_eff: must be in (0, 2]")
        for f in ("compute_eff", "mem_eff", "net_eff", "capacity_usable"):
            if not 0 < getattr(self, f) <= 1:
                raise SpecError(f"{f}: must be in (0, 1]")
        for k, v in self.gemm_eff_by_tokens.items():
            if not (float(k) >= 1 and 0 < v <= 2):
                raise SpecError(f"gemm_eff_by_tokens.{k}: need tokens >= 1 and 0 < eff <= 2")
        for f in ("link_gbps", "kernel_overhead_us", "link_latency_us", "tier2_gb", "tier2_gbps"):
            if getattr(self, f) < 0:
                raise SpecError(f"{f}: must be >= 0")

    def gemm_eff(self, tokens: float) -> float:
        """Calibrated matmul efficiency at M = `tokens`, log-linear between measured points."""
        if not self.gemm_eff_by_tokens or tokens <= 0:
            return 1.0
        pts = sorted((float(k), v) for k, v in self.gemm_eff_by_tokens.items())
        if tokens <= pts[0][0]:
            return pts[0][1]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False):
            if tokens <= x1:
                f = math.log(tokens / x0) / math.log(x1 / x0)
                return y0 + f * (y1 - y0)
        return pts[-1][1]

    def peak(self, dtype: str) -> float:
        """Peak FLOP/s for a compute dtype."""
        if dtype not in self.peak_tflops:
            raise SpecError(f"peak_tflops.{dtype}: {self.name} has no {dtype} tensor throughput")
        return self.peak_tflops[dtype] * 1e12


@dataclass(frozen=True)
class Workload:
    batch: int = 1
    prompt_len: int = 1024
    output_len: int = 256
    weight_dtype: str = "bf16"
    kv_dtype: str = "bf16"
    act_dtype: str = "bf16"  # also the tensor-core compute dtype
    tp: int = 1

    def __post_init__(self):
        for f in ("batch", "prompt_len", "output_len", "tp"):
            _positive(self, f)
        for f in ("weight_dtype", "kv_dtype", "act_dtype"):
            if getattr(self, f) not in DTYPE_BYTES:
                raise SpecError(f"{f}: unknown dtype {getattr(self, f)!r}")

    def check(self, model: ModelSpec) -> None:
        """Cross-validate against a model (tensor parallelism must split heads evenly)."""
        if model.n_heads % self.tp:
            raise SpecError(f"tp: {self.tp} does not divide n_heads={model.n_heads}")
        if model.mla is None and model.n_kv_heads % self.tp and self.tp % model.n_kv_heads:
            raise SpecError(f"tp: {self.tp} incompatible with n_kv_heads={model.n_kv_heads}")


@dataclass(frozen=True)
class Twin:
    """Tiny executable dense model used for empirical validation."""

    model: ModelSpec
    batches: tuple[int, ...] = field(default=())
    contexts: tuple[int, ...] = field(default=())
    max_kv_tokens: int = 65536  # cap on batch * context to bound device memory
    dtype: str = "fp16"
    schema_version: int = SCHEMA_VERSION


def _positive(obj, name, prefix=""):
    if not getattr(obj, name) > 0:
        raise SpecError(f"{prefix}{name}: must be > 0")


def _build(cls, data: dict, where: str):
    if not isinstance(data, dict):
        raise SpecError(f"{where}: expected an object")
    names = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(names))
    if unknown:
        raise SpecError(f"{where}{unknown[0]}: unknown field")
    kwargs = {}
    for k, v in data.items():
        if k in ("moe", "mla") and v is not None:
            v = _build(MoE if k == "moe" else MLA, v, f"{k}.")
        elif k == "model" and cls is Twin:
            v = _build(ModelSpec, v, "model.")
        elif isinstance(v, list):
            v = tuple(v)
        kwargs[k] = v
    if kwargs.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise SpecError(f"{where}schema_version: expected {SCHEMA_VERSION}")
    try:
        return cls(**kwargs)
    except TypeError as e:  # missing required field
        raise SpecError(f"{where}{e}") from None


def _read_json(path: Path) -> dict:
    if path.stat().st_size > MAX_SPEC_BYTES:
        raise SpecError(f"{path.name}: spec file exceeds {MAX_SPEC_BYTES} bytes")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise SpecError(f"{path.name}: invalid JSON ({e})") from None


def _load(kind: str, cls, ref: str):
    """Load a preset by name or a spec from a JSON file path."""
    path = Path(ref)
    if not path.suffix == ".json":
        path = PRESETS / kind / f"{ref}.json"
        if not path.is_file():
            raise SpecError(f"{kind}: unknown preset {ref!r}; available: {preset_names(kind)}")
    return _build(cls, _read_json(path), "")


def load_model(ref: str) -> ModelSpec:
    return _load("models", ModelSpec, ref)


def load_hardware(ref: str) -> HardwareSpec:
    return _load("hardware", HardwareSpec, ref)


def load_twin(ref: str) -> Twin:
    return _load("twins", Twin, ref)


def hardware_from_dict(data: dict) -> HardwareSpec:
    return _build(HardwareSpec, data, "")


def preset_names(kind: str) -> list[str]:
    return sorted(p.stem for p in (PRESETS / kind).glob("*.json"))


def to_dict(spec) -> dict:
    return dataclasses.asdict(spec)


def spec_hash(spec) -> str:
    blob = json.dumps(to_dict(spec), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def with_overrides(hw: HardwareSpec, overrides: dict[str, float]) -> HardwareSpec:
    """Apply `field=value` overrides (e.g. from the CLI); peak dtypes as `peak_tflops.fp8`."""
    peaks = dict(hw.peak_tflops)
    plain = {}
    for k, v in overrides.items():
        if k.startswith("peak_tflops."):
            try:
                peaks[k.split(".", 1)[1]] = float(v)
            except (TypeError, ValueError):
                raise SpecError(f"{k}: expected a number, got {v!r}") from None
        elif k == "overhead_mode":
            plain[k] = str(v)
        elif isinstance(getattr(hw, k, None), int | float) and k != "schema_version":
            try:
                plain[k] = float(v)
            except (TypeError, ValueError):
                raise SpecError(f"{k}: expected a number, got {v!r}") from None
        else:
            raise SpecError(f"{k}: not an overridable hardware field")
    return dataclasses.replace(hw, peak_tflops=peaks, **plain)


def parse_sets(items: list[str]) -> dict[str, str]:
    """Parse repeated `FIELD=VALUE` CLI options."""
    pairs = [s.split("=", 1) for s in items]
    if any(len(p) != 2 for p in pairs):
        raise SpecError("--set: expected FIELD=VALUE")
    return dict(pairs)
