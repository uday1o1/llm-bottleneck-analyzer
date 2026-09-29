"""Operator graph: FLOPs and HBM bytes of every op in a prefill pass or one decode step.

Conventions: a multiply-add is 2 FLOPs; quantities are per tensor-parallel rank; `count` is how
many layers repeat an op; `kernels` follows the eager PyTorch reference implementation used by the
executable twin (15 launches per dense layer), and HBM traffic is compulsory traffic. See
docs/methodology.md for the equations and their assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass

from .specs import DTYPE_BYTES, ModelSpec, Workload

PREFILL, DECODE = "prefill", "decode"


@dataclass(frozen=True)
class Op:
    name: str
    kind: str  # gemm | attention | elementwise | moe | comm
    flops: float = 0.0
    hbm_bytes: float = 0.0
    weight_bytes: float = 0.0  # part of hbm_bytes that is parameters
    kv_bytes: float = 0.0  # KV-cache traffic (included in hbm_bytes + tier2_bytes)
    tier2_bytes: float = 0.0  # KV read from the capacity tier (not in hbm_bytes)
    comm_bytes: float = 0.0  # all-reduce payload (comm ops only)
    ranks: int = 1
    tokens: float = 0.0  # matmul row count M (for calibrated shape efficiency; 0 = not a matmul)
    kernels: int = 1  # launches in a standard unfused implementation
    count: int = 1  # layers that repeat this op


# ---------------------------------------------------------------- parameters


def attn_weights(m: ModelSpec) -> dict[str, int]:
    """Attention parameter elements per layer (unsharded), keyed by projection."""
    d = m.d_model
    if m.mla:
        a = m.mla
        return {
            "q_a": d * a.q_lora_rank,
            "q_b": a.q_lora_rank * m.n_heads * (a.qk_nope_head_dim + a.qk_rope_head_dim),
            "kv_a": d * (a.kv_lora_rank + a.qk_rope_head_dim),
            "kv_b": a.kv_lora_rank * m.n_heads * (a.qk_nope_head_dim + a.v_head_dim),
            "o": m.n_heads * a.v_head_dim * d,
            "norms": a.q_lora_rank + a.kv_lora_rank,
        }
    return {"qkv": d * (m.n_heads + 2 * m.n_kv_heads) * m.head_dim, "o": m.n_heads * m.head_dim * d}


def n_moe_layers(m: ModelSpec) -> int:
    return m.n_layers - m.moe.first_dense if m.moe else 0


def param_counts(m: ModelSpec) -> tuple[int, int]:
    """(total, active-per-token) parameter elements."""
    d = m.d_model
    per_layer_common = sum(attn_weights(m).values()) + 2 * d
    embed = m.vocab * d * (1 if m.tied_embeddings else 2) + d
    dense_mlp = 3 * d * m.d_ff
    n_moe = n_moe_layers(m)
    total = active = embed + m.n_layers * per_layer_common + (m.n_layers - n_moe) * dense_mlp
    if m.moe:
        e = m.moe
        expert = 3 * d * e.d_ff_expert
        total += n_moe * (d * e.n_experts + (e.n_experts + e.n_shared) * expert)
        active += n_moe * (d * e.n_experts + (e.top_k + e.n_shared) * expert)
    return total, active


def expected_distinct_experts(n_experts: int, top_k: int, tokens: int) -> float:
    """Expected experts touched by `tokens` tokens under uniform independent top-k routing."""
    return n_experts * (1.0 - (1.0 - top_k / n_experts) ** tokens)


def kv_bytes_per_token(m: ModelSpec, wl: Workload) -> float:
    """KV-cache bytes per token per rank, all layers. MLA latents are replicated across ranks."""
    b = DTYPE_BYTES[wl.kv_dtype]
    if m.mla:
        return m.n_layers * (m.mla.kv_lora_rank + m.mla.qk_rope_head_dim) * b
    return m.n_layers * 2 * kv_heads_per_rank(m, wl.tp) * m.head_dim * b


def kv_heads_per_rank(m: ModelSpec, tp: int) -> int:
    return max(1, m.n_kv_heads // tp)  # KV heads replicate when tp > n_kv_heads


def weight_bytes_per_rank(m: ModelSpec, wl: Workload) -> float:
    """Stored parameter bytes on one rank (all experts), honoring TP sharding and replication."""
    total, _ = param_counts(m)
    d, tp = m.d_model, wl.tp
    replicated = (2 * m.n_layers + 1) * d  # norms
    if m.mla:
        w = attn_weights(m)
        replicated += m.n_layers * (w["q_a"] + w["kv_a"] + w["norms"])
    if m.moe:
        replicated += n_moe_layers(m) * d * m.moe.n_experts  # routers
    extra = 0.0
    if not m.mla:  # K/V projections replicate with their heads when tp > n_kv_heads
        extra = m.n_layers * d * 2 * m.head_dim * (kv_heads_per_rank(m, tp) - m.n_kv_heads / tp)
    return ((total - replicated) / tp + replicated + extra) * DTYPE_BYTES[wl.weight_dtype]


# ---------------------------------------------------------------- graph


def build_ops(
    m: ModelSpec, wl: Workload, phase: str, context: int = 0, tier2_frac: float = 0.0
) -> list[Op]:
    """Ops for a prefill pass over `wl.prompt_len` tokens per sequence, or one decode step at
    KV length `context` (tokens already cached plus the new one). `tier2_frac` of decode KV reads
    come from the capacity tier."""
    if phase not in (PREFILL, DECODE):
        raise ValueError(f"phase: {phase!r}")
    B, tp, d = wl.batch, wl.tp, m.d_model
    S = wl.prompt_len if phase == PREFILL else 1
    T = B * S  # new tokens in this pass
    bw, ba = DTYPE_BYTES[wl.weight_dtype], DTYPE_BYTES[wl.act_dtype]
    L = m.n_layers
    ops: list[Op] = []

    def gemm(name, k, n, count, tokens=T, residual=False):
        """GEMM with per-rank reduction dim k and output dim n; `residual` adds the following
        residual-add launch and its traffic."""
        w = k * n
        extra = 3 * tokens * n * ba if residual else 0
        ops.append(Op(name, "gemm", 2 * tokens * w, w * bw + tokens * (k + n) * ba + extra,
                      weight_bytes=w * bw, kernels=2 if residual else 1, count=count,
                      tokens=tokens))

    def norm(name, count):
        ops.append(Op(name, "elementwise", 5 * T * d, 2 * T * d * ba + d * bw, count=count))

    def allreduce(name, count):
        if tp > 1:
            ops.append(Op(name, "comm", comm_bytes=T * d * ba, ranks=tp, count=count))

    ops.append(Op("embed", "elementwise", 0, 2 * T * d * ba))
    norm("attn_norm", L)
    if m.mla:
        _mla_ops(m, wl, phase, context, T, gemm, ops, tier2_frac)
    else:
        _gqa_ops(m, wl, phase, context, T, gemm, ops, tier2_frac)
    allreduce("attn_allreduce", L)
    norm("mlp_norm", L)
    n_moe = n_moe_layers(m)
    if L - n_moe:
        _mlp(ops, "mlp", T, d, m.d_ff / tp, bw, ba, L - n_moe)
    if n_moe:
        e = m.moe
        gemm("moe_router", d, e.n_experts, n_moe)
        distinct = expected_distinct_experts(e.n_experts, e.top_k, T)
        f_r = e.d_ff_expert / tp
        w = distinct * 3 * d * f_r * bw
        act = T * e.top_k * (2 * d + 6 * f_r) * ba + 3 * T * d * ba
        ops.append(Op("moe_experts", "moe", 2 * T * e.top_k * 3 * d * f_r, w + act,
                      weight_bytes=w, kernels=5, count=n_moe, tokens=T * e.top_k / distinct))
        if e.n_shared:
            _mlp(ops, "moe_shared", T, d, e.n_shared * f_r, bw, ba, n_moe, residual=False)
    allreduce("mlp_allreduce", L)
    norm("final_norm", 1)
    gemm("lm_head", d, m.vocab / tp, 1, tokens=B)  # logits only for the last position per sequence
    return ops


def _mlp(ops, name, T, d, f_r, bw, ba, count, residual=True):
    """SwiGLU MLP: fused gate/up GEMM, SiLU, multiply, down GEMM, then the residual add."""
    w = 3 * d * f_r
    act = T * (2 * d + 8 * f_r + (3 * d if residual else 0)) * ba
    ops.append(Op(name, "gemm", 2 * T * w, w * bw + act, weight_bytes=w * bw,
                  kernels=5 if residual else 4, count=count, tokens=T))


def _attention(ops, flops, other_bytes, kv_read, kv_write, tier2_frac, weight_bytes=0.0,
               kernels=5, count=1, rows=1.0):
    """K write, V write, QK^T, softmax, AV (eager); traffic is the compulsory (fused) traffic.
    `rows` is the query rows per KV head, the M of the batched score matmul."""
    t2 = kv_read * tier2_frac
    ops.append(Op("attention", "attention", flops, other_bytes + kv_read - t2 + kv_write,
                  weight_bytes=weight_bytes, kv_bytes=kv_read + kv_write, tier2_bytes=t2,
                  kernels=kernels, count=count, tokens=rows))


def _gqa_ops(m, wl, phase, context, T, gemm, ops, tier2_frac):
    B, tp, d, hd = wl.batch, wl.tp, m.d_model, m.head_dim
    h, hkv = m.n_heads // tp, kv_heads_per_rank(m, tp)
    ba, bkv = DTYPE_BYTES[wl.act_dtype], DTYPE_BYTES[wl.kv_dtype]
    L = m.n_layers
    gemm("qkv_proj", d, (h + 2 * hkv) * hd, L)
    kv_write = T * 2 * hkv * hd * bkv
    qo = T * 2 * h * hd * ba
    if phase == DECODE:
        kv_read = B * context * 2 * hkv * hd * bkv
        _attention(ops, 4 * B * context * h * hd, qo, kv_read, kv_write, tier2_frac, count=L,
                   rows=h / hkv)
    else:  # compulsory traffic only: Q, K, V read once, O and the KV cache written once
        S = wl.prompt_len
        flops = 4 * B * h * hd * S * (S + 1) / 2
        _attention(ops, flops, qo + T * 2 * hkv * hd * ba, 0.0, kv_write, 0.0, count=L,
                   rows=S * h / hkv)
    gemm("o_proj", h * hd, d, L, residual=True)  # row-parallel: reduction dim is sharded


def _mla_ops(m, wl, phase, context, T, gemm, ops, tier2_frac):
    """DeepSeek-style MLA. Decode absorbs W_UK/W_UV into the query/output path and attends over
    the latent cache; prefill materializes per-head K and V (the non-absorbed form)."""
    a, B, tp, d = m.mla, wl.batch, wl.tp, m.d_model
    h = m.n_heads // tp
    bw, ba, bkv = (DTYPE_BYTES[x] for x in (wl.weight_dtype, wl.act_dtype, wl.kv_dtype))
    L = m.n_layers
    lat = a.kv_lora_rank + a.qk_rope_head_dim
    gemm("q_a_proj", d, a.q_lora_rank, L)  # replicated across ranks
    gemm("q_b_proj", a.q_lora_rank, h * (a.qk_nope_head_dim + a.qk_rope_head_dim), L)
    gemm("kv_a_proj", d, lat, L)  # replicated across ranks
    kv_write = T * lat * bkv
    if phase == DECODE:
        w_absorb = a.kv_lora_rank * h * (a.qk_nope_head_dim + a.v_head_dim)  # W_UK and W_UV
        kv_read = B * context * lat * bkv
        flops = (2 * T * w_absorb
                 + 2 * B * context * h * lat + 2 * B * context * h * a.kv_lora_rank)
        _attention(ops, flops, w_absorb * bw + T * h * (lat + a.kv_lora_rank) * ba, kv_read,
                   kv_write, tier2_frac, weight_bytes=w_absorb * bw, kernels=7, count=L,
                   rows=h)
    else:
        S = wl.prompt_len
        gemm("kv_b_proj", a.kv_lora_rank, h * (a.qk_nope_head_dim + a.v_head_dim), L)
        qk = a.qk_nope_head_dim + a.qk_rope_head_dim
        flops = 2 * B * h * (qk + a.v_head_dim) * S * (S + 1) / 2
        io = T * h * (2 * qk + 2 * a.v_head_dim) * ba
        _attention(ops, flops, io, 0.0, kv_write, 0.0, kernels=6, count=L, rows=S)
    gemm("o_proj", h * a.v_head_dim, d, L, residual=True)
