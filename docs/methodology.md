# Methodology

This document states every equation the analyzer uses, with its assumptions.
Code references point to `src/llmba/`.

## Notation

`B` batch (sequences), `S` prompt length, `O` output length, `C` KV length at a decode step, `T` new tokens in a pass (`B*S` for prefill, `B` for decode), `d` model width, `H` query heads, `Hkv` KV heads, `hd` head dimension, `d_ff` MLP width, `L` layers, `V` vocabulary, `tp` tensor-parallel degree, `Hkv_r = max(1, Hkv/tp)`.
Byte widths `b_w`, `b_a`, `b_kv` come from the weight, activation, and KV dtypes (`fp32` 4, `bf16`/`fp16` 2, `fp8` 1).
A multiply-add is 2 FLOPs.
Units are encoded in field names: `*_gb` decimal gigabytes, `*_gbps` 1e9 bytes per second, `*_tflops` 1e12 FLOP/s dense, `*_us` microseconds.

## Parameter accounting (`ops.py: param_counts`)

Per layer: attention projections (`qkv = d*(H+2*Hkv)*hd`, `o = H*hd*d`, or the five MLA projections) plus two RMSNorm vectors; dense MLP `3*d*d_ff` (SwiGLU); MoE layers add a router `d*E` and `(E + n_shared)` experts of `3*d*d_ff_expert`.
Embeddings `V*d` (twice when untied) plus the final norm.
Active parameters replace `E` by `top_k` in expert counts.
The same weight definitions drive the operator graph, so the parameter golden test also checks the bytes the cost engine streams.

## Operator graph (`ops.py: build_ops`)

Each layer emits, per tensor-parallel rank:

| op | FLOPs | HBM bytes | launches |
| --- | --- | --- | --- |
| `attn_norm`, `mlp_norm` | `5*T*d` | `2*T*d*b_a + d*b_w` | 1 |
| `qkv_proj` | `2*T*d*(H/tp + 2*Hkv_r)*hd` | weights + `T*(k+n)*b_a` | 1 |
| `attention` (decode) | `4*B*C*(H/tp)*hd` | KV read `B*C*2*Hkv_r*hd*b_kv` + KV write + Q/O | 5 |
| `attention` (prefill) | `4*B*(H/tp)*hd*S*(S+1)/2` | Q, K, V read once, O and KV written once | 5 |
| `o_proj` + residual add | `2*T*(H/tp)*hd*d` | weights + activations + `3*T*d*b_a` | 2 |
| `mlp` (SwiGLU) + residual add | `6*T*d*d_ff/tp` | weights + `T*(2d + 8*d_ff/tp + 3d)*b_a` | 5 |
| `attn_allreduce`, `mlp_allreduce` | 0 | 0 (network: `T*d*b_a`) | 1 |

Model-level ops: `embed` (gather), `final_norm`, `lm_head` (logits for the last position of each sequence only).
Launch counts follow the eager PyTorch reference implementation used by the executable twin (15 per dense layer, plus one per all-reduce when `tp > 1`); attention bytes are compulsory (FlashAttention-style) traffic.

### MoE

Under uniform independent top-k routing, `T` tokens touch `E_u = E*(1 - (1 - k/E)^T)` distinct experts in expectation.
Expert weight bytes are `E_u*3*d*d_ff_expert*b_w/tp`; FLOPs are `6*T*k*d*d_ff_expert/tp`.
Skewed routing touches fewer experts and anti-correlated routing can touch more, so this is an assumption, not a bound.
Because every touched expert must be streamed while each only sees about `T*k/E_u` tokens, an MoE layer's weight-GEMM knee sits roughly `E/k` times above a dense layer's until all experts are active.

### MLA (DeepSeek-V3)

The latent KV cache holds `(r_kv + d_rope)*b_kv` bytes per token per layer and is replicated on every rank.
Decode absorbs `W_UK` and `W_UV`: attention FLOPs `2*B*C*(H/tp)*(2*r_kv + d_rope)` plus the absorbed projections; KV bytes `B*C*(r_kv + d_rope)*b_kv`.
Prefill uses the non-absorbed form with `kv_b` materializing per-head K and V.

## Cost model (`cost.py`)

For each op:

- `t_compute = FLOPs / (peak[act_dtype] * compute_eff)`
- `t_hbm = hbm_bytes / (hbm_gbps * 1e9 * mem_eff)`
- `t_tier2 = tier2_bytes / (tier2_gbps * 1e9 * mem_eff)`
- `t_network = 2*(p-1)/p * bytes / (link_gbps * 1e9 * net_eff) + 2*(p-1)*link_latency` (ring all-reduce)
- `t_busy = max(t_compute, t_hbm, t_tier2, t_network) / eff`, where `eff` is the calibrated shape efficiency (`gemm_eff_by_tokens` at the op's matmul row count, `attn_eff` for attention, 1.0 when uncalibrated)
- `t_overhead = launches * kernel_overhead_us`

`additive` mode: step time `= sum(count * (t_busy + t_overhead))`.
`overlap` mode: step time `= max(sum busy, sum overhead)`.
The main resource of an op is the argmax of its busy components; the bound class is `overhead` when overhead exceeds busy time.
Ties break in the order compute, hbm, tier2, network, overhead.

## Requests (`analyze.py`)

TTFT is the prefill pass.
TPOT is the mean decode step over positions `S+1 .. S+O`, evaluated exactly for `O <= 64` and otherwise at the midpoints of 64 strata (tested within 0.5 percent of the exact sum).
Output throughput for a static batch is `B*O / (TTFT + O*TPOT)`.

## Capacity and the capacity tier

Weights must fit in `hbm_gb*1e9*capacity_usable` (default 0.9 absorbs activation workspace).
KV (sized at the final length `S+O` per sequence) fills the remaining HBM, then spills to the capacity tier (`tier2_gb`).
Decode attention reads the spilled fraction at `tier2_gbps`, concurrently with HBM (memory time is the maximum of the two tiers).

## Elasticity

Elasticity of time `T` with respect to making resource `r` faster by factor `f` is `e_r = -d ln T / d ln f`.
In `additive` mode `T` is a sum of maxima, so away from ties `e_r` equals the share of `T` in which `r` is an op's main resource, and overhead contributes its additive share.
In `overlap` mode each step is all-or-nothing between busy and overhead time.
A faster network scales link bandwidth up and link latency down together.
A test compares these analytic shares with central finite differences on 200 seeded operating points across both overhead modes and with and without a capacity tier.

Two objectives are reported and never mixed:

- Latency elasticity: end-to-end request time at a fixed batch (the analytic shares above).
- Throughput elasticity: output tokens/s at the best capacity-feasible real-valued batch (capped at 512 concurrent sequences), by finite differences for compute, HBM bandwidth, network, overhead, and HBM capacity.
  A capacity elasticity above 1 means weights occupy most of HBM, so each percent of capacity adds more than a percent of KV room.

## Intercepts (`intercepts.py`)

- Weight-GEMM knee: smallest batch at which weight GEMMs (and MoE experts) are mostly compute-bound, by exponential search and bisection; reported within the capacity-feasible range and unconstrained.
- Whole-step knee: smallest batch at which compute-bound time is at least half of the decode step; GQA attention's arithmetic intensity is fixed at `2*H/(Hkv*b_kv)`, so this is often `INCONCLUSIVE`, which is a finding.
- Analytic GEMM ridge: `(peak*compute_eff) / (hbm_gbps*1e9*mem_eff) * b_w / 2`.
- Attention-dominance context: smallest KV length at which attention is at least half of the decode step.
- Capacity wall: maximum batch versus context.
- TP saturation: smallest TP degree beyond which the next degree improves TPOT by less than 5 percent (degrees that do not fit are skipped).
- Capacity-tier break-even: smallest tier bandwidth at which a tier of the given size raises capacity-limited throughput over HBM alone.

## Scheduler-independent throughput ceiling (claim S3)

Every decode step streams all non-embedding weights `W_read` once plus the KV of each resident sequence, and resident KV never exceeds `Cap - W_res`.
Summing over steps, each request moves at least `M_req = KV_req * (1 + W_read / (Cap - W_res))` bytes with `KV_req = kv_tok * (O-1) * (S + (O-2)/2)`.
Then `t_req >= max(F_req / P, M_req / BW)` and aggregate output throughput is at most `O / t_req`, for any batching schedule.
Every choice raises the ceiling: binary-GiB capacity, every FLOP at the best dense peak, 1-byte weights and KV, efficiency 1, no overhead.

## Calibration and the executable twin (claim P1)

`llmba measure calibrate` runs only standalone microbenchmarks:

- copy bandwidth (read plus write bytes);
- a 4096-cubed fp16 GEMM for peak throughput;
- a 200-launch loop of tiny in-place adds for per-launch overhead;
- a matmul row-count sweep (M = 1 to 256) on two generic weight shapes, timed as 20 back-to-back calls, giving `gemm_eff_by_tokens`;
- a decode-attention sweep on generic shapes, giving `attn_eff`.

`llmba measure validate` builds an eager PyTorch dense GQA decoder with random weights (SwiGLU, RMSNorm, no RoPE), times one decode step per grid point (3 warm-up and 15 timed iterations, device synchronization at the boundary, seeded shuffled order, median), and grades predictions with the pre-registered rules in `BUILD_PLAN.md` section 13.
No knob is fitted to twin measurements.

## Limitations

- First-order analytical model: compulsory HBM traffic, no on-chip cache model, no cycle accuracy.
- Datacenter GPU numbers are model outputs from datasheet specs with assumed default efficiencies (compute 0.7, memory 0.8, network 0.8, 4 us per launch); they are not measurements.
- Throughput what-ifs assume static batching.
- MoE assumes uniform independent routing.
- The empirical check runs on an Apple M2 GPU with LPDDR unified memory; it validates the methodology on real hardware, not HBM or NVIDIA accuracy.
