# LLMBottleneckAnalyzer: AI Workload Characterization and Critical Path Analysis

## 1. Product definition

LLMBottleneckAnalyzer (`llmba`) is a local-first Python tool that answers one question for an LLM inference workload:
which hardware resource (compute, HBM bandwidth, HBM capacity, interconnect, or per-kernel overhead) is on the critical path, by how much, and at what batch size, context length, or parallelism degree does that answer change.

Primary user: a performance or memory-system architect (or an inference engineer) who must decide which knob of a current or next-generation accelerator memory system matters for an emerging LLM workload class before any silicon or cluster exists.

Triggering problem: public LLM inference discussions quote "decode is memory-bound" as folklore.
Architects need the operating-point-specific answer: where the memory-to-compute crossover (the intercept) sits for GQA, MLA, and MoE models, how long-context and long-reasoning workloads shift it, and the marginal value of more bandwidth versus more capacity versus more FLOPs.

Inputs: a model architecture spec (JSON), a hardware spec (JSON), and a workload (batch, prompt length, output length, dtypes, tensor-parallel degree), or a study file that names grids of these.

Outputs: per-operator characterization (FLOPs, bytes per memory level, arithmetic intensity, bound class), per-phase and per-request critical-path attribution (TTFT, TPOT, throughput), capacity limits, crossover intercepts, resource elasticities, ranked what-if recommendations, a synthesized characterization dataset with a content-addressed manifest, and a Markdown report with figures.

Owned technical center:

1. An analytical operator-graph cost model covering dense GQA, MLA (latent KV), and MoE (expected-distinct-expert weight traffic) architectures, tensor-parallel ring all-reduce, and a two-tier memory system (HBM plus an optional capacity tier such as LPDDR or CXL-attached memory that holds KV spilled past the HBM budget).
2. Intercept analysis: solving for memory-to-compute crossover batch, attention-dominance context length, capacity wall, and tensor-parallel saturation, and computing exact resource elasticities as critical-path time shares.
3. A closed-loop empirical check: calibrate a hardware profile from microbenchmarks on a real device, then predict an out-of-sample executable transformer decode step on the same device and grade the prediction with predeclared gates and a negative control.

Primary falsifiable claim (P1): with a hardware profile calibrated only from microbenchmarks, the model predicts the measured decode-step latency of an executable transformer twin on the local Apple M2 GPU with a median absolute percentage error at or below the proposed threshold in section 13, and predicts the measured knee batch (the 1.5x step-time rule of section 13) within one power-of-two grid step, on a final grid that is not used during development.

Secondary claims:

- S1: architecture accounting is correct: computed parameter counts match officially published totals within 1 percent for every bundled model, and KV-cache bytes per token match closed-form values exactly.
- S2: analytic elasticities equal finite-difference elasticities within tolerance away from ties (a metamorphic consistency property of the cost model).
- S3: no officially published measured throughput row used as a reference point exceeds the model's scheduler-independent throughput ceiling (section 13) for the same configuration.
- S4: the characterization dataset and `results.json` are byte-reproducible from the same inputs (figures are excluded from the byte claim).

Prohibited claims: no claim of NVIDIA, AMD, or HBM-measured accuracy (all GPU and HBM numbers are model outputs from datasheet specs); no claim of cycle accuracy; no claim that recommendations are optimal; no claim that Apple M2 results transfer quantitatively to datacenter GPUs; no claim of serving-system (scheduler, queueing) fidelity.

## 2. Primary user workflows and demonstration path

Workflow A (single operating point):
`llmba analyze --model llama-3.1-70b --hw h100-sxm --tp 4 --batch 64 --prompt 2048 --output 512` prints the phase table, top critical-path operators, bound classes, capacity headroom, and elasticities.

Workflow B (characterization sweep and data synthesis):
`llmba sweep --models llama-3.1-8b,mixtral-8x7b --hw a100-80gb,h100-sxm,h200 --batch 1,8,64,256 --prompt 1024 --output 256 --out results/sweep` writes `dataset.csv` (rows that exceed capacity are kept and flagged `fits=false`) and `manifest.json`.

Workflow C (intercepts and recommendations):
`llmba intercepts --model deepseek-v3 --hw h200 --tp 8 --weight-dtype fp8` and `llmba recommend --study studies/next_gen_memory.json` rank memory-system and software intercepts per workload archetype.

Workflow D (full report, the demonstration):
`llmba report --study studies/next_gen_memory.json --out results/report` writes `report.md`, `results.json`, and figures.

Workflow E (empirical closed loop, local hardware):
`llmba measure calibrate --device mps --out results/calib` then `llmba measure validate --profile results/calib/profile.json --twin tiny-dev --out results/validation`.

Interview demonstration (under five minutes): run Workflow D, open the roofline and intercept-map figures, show that MoE shifts the decode knee, show the throughput-elasticity ranking moving from bandwidth toward capacity for long-context workloads and the tier-2 bandwidth intercept, then show the Workflow E validation table and its negative control failing as designed.

## 3. Hiring evidence and public-source provenance

The project brief was supplied by the owner as a fixed concept targeting GPU architecture, memory-system, and AI-performance-modeling roles; no private or enterprise evidence is used.
Relevant public role families include GPU performance architect, memory-system or HBM architect, AI workload characterization engineer, and inference performance engineer at accelerator and memory vendors.
Mapping (an inference from typical public job descriptions, not a guarantee):

| Hiring signal | Project proof artifact |
| --- | --- |
| Performance modeling and roofline analysis | Operator-graph cost engine, hierarchical roofline figure, elasticity math with tests |
| Memory hierarchy and HBM understanding | HBM plus capacity-tier model, KV-cache capacity wall, bandwidth versus capacity what-ifs, tier-2 intercept |
| Workload characterization | Archetype studies, synthesized dataset with manifest, per-op bound classification |
| Measurement discipline | Microbenchmark calibration, out-of-sample twin validation, negative control, holdout grid |
| Python engineering | Typed package, CLI, deterministic outputs, tests, clean-checkout workflow |

## 4. Portfolio differentiation and prior art

Close predecessors (public, open source or published):

- LLM-Viewer (roofline analysis of LLM layers): per-layer roofline for dense models.
- LLMCompass and GenZ: hardware evaluation frameworks for LLM inference.
- llm-analysis: latency and memory estimates for training and inference.
- Vidur: simulation of LLM serving systems with profiled operators.

What this project reuses: the standard roofline idea and public architecture configs.
What it owns: (a) intercept solving and exact elasticity attribution as critical-path time shares, (b) MoE expected-distinct-expert traffic and MLA latent-cache accounting in the same engine as GQA, (c) a calibrated predict-then-measure loop with a holdout grid and negative control on real hardware the owner has.
The evaluation distinguishes this from a reimplementation because claims P1, S2, and S3 are tested, not asserted.

## 5. Scope

### Milestone 0 feasibility (complete before core work)

Prove toolchain, spec schema, parameter accounting for one model, one end-to-end prediction, and the MPS measurement feasibility spike.

### Portfolio-ready core (M1 to M6)

Specs and presets, operator graph and cost engine, CLI analyze, sweeps with manifest, intercepts and recommendations, report with figures.

### Extended V1 (M7 to M9)

Local calibration and twin validation, published-ceiling check, final holdout run, documentation, and clean-checkout acceptance.

### Follow-on releases (not V1 gates)

- Follow-on F1: CUDA qualification of the twin on Colab Pro (same `llmba measure` entry point on a CUDA device).
- Follow-on F2: pipeline and expert parallelism, speculative decoding, and a continuous-batching scheduler simulator.
- Follow-on F3: MoE and MLA executable twins.

### Non-goals

V1 has no serving system, no scheduler simulation (follow-on F2), no training workloads, no cycle-level simulation, no on-chip cache model, no web UI, no cloud deployment, and no network services.

## 6. System architecture and component ownership

All components are Python in one package `llmba`; the analytical engine uses only the standard library.

| Component | Module | Responsibility | Inputs | Outputs | Failure modes |
| --- | --- | --- | --- | --- | --- |
| Specs | `specs.py` | Load and validate model, hardware, workload specs; presets | JSON files, preset names | Frozen dataclasses | `SpecError` with field name |
| Operator graph | `ops.py` | Build per-layer and per-phase op lists with FLOPs and bytes | Specs, phase, tokens, context | `Op` list | `SpecError` for unsupported combos |
| Cost engine | `cost.py` | Hierarchical roofline time, bound class, overhead, comm | Ops, hardware | `OpCost` list, phase totals | none (pure math, validated inputs) |
| Analysis | `analyze.py` | Prefill, decode, request metrics, capacity, critical path, elasticities | Specs, workload | `Analysis` dict | capacity overflow flagged, not raised |
| Sweep | `sweep.py` | Grid expansion, dataset rows, CSV, manifest | Grid definition | `dataset.csv`, `manifest.json` | refuses to overwrite existing run dir |
| Intercepts | `intercepts.py` | Crossovers, what-ifs, recommendations | Specs, archetypes | Tables and text | `INCONCLUSIVE` when no crossover in range |
| Report | `report.py` | Markdown, JSON, matplotlib figures | Study | Report dir | missing matplotlib gives clear error |
| Measure | `measure.py` | Microbenchmarks, executable twin, validation gate | Device, twin preset | Profile JSON, validation evidence | device unavailable gives `BLOCKED` |
| CLI | `cli.py` | argparse front end | argv | stdout, files, exit codes | usage errors exit 2 |

Data flow: specs -> ops -> cost -> analyze -> (sweep | intercepts | report); measure produces a hardware profile consumed by the same cost engine.

## 7. Canonical data, schema, and unit contracts

Schema version: every spec JSON and output JSON carries `"schema_version": 1`.
Producer version: outputs carry `llmba_version` from the package.

Units (field names encode units; no field permits another unit):

- Bytes are bytes; `*_gb` fields are decimal gigabytes (1e9 bytes) as printed on vendor datasheets; `*_gbps` is 1e9 bytes per second.
- `*_tflops` is 1e12 floating-point operations per second, dense (non-sparse) tensor throughput.
- `*_us` is microseconds; internal times are seconds (float).
- One multiply-add counts as 2 FLOPs.
- Interconnect `link_gbps` is per GPU, per direction.
- Dtypes are named `fp32`, `bf16`, `fp16`, `fp8`, with byte sizes 4, 2, 2, 1.

Identity: presets are addressed by kebab-case `name`; each preset records `source_urls`.
A spec hash is SHA-256 of its canonical JSON (sorted keys, compact separators).

Missing states: optional architecture blocks (`moe`, `mla`) are absent or null when not applicable; any unknown key is a `SpecError` (typo protection).
Hardware efficiency knobs (`compute_eff`, `mem_eff`, `net_eff`, `kernel_overhead_us`) are explicit calibration knobs with documented defaults, not datasheet values.

Deterministic serialization: JSON with `sort_keys=True`, `indent=2`, trailing newline; CSV rows in grid order with floats formatted by `format(x, ".9g")`.

## 8. Algorithms

### 8.1 Operator graph

For a forward pass processing `T` new tokens with per-sequence context `C` (decode: `T = B`, `C = position`; prefill: `T = B*S`, causal over `S`), each layer emits ops.
Per-rank quantities divide head counts and FFN widths by the tensor-parallel degree `tp`.

- `attn_norm`, `mlp_norm`: elementwise; bytes `2*T*d*b_act`.
- `qkv_proj` (GQA): weight `d*(H/tp + 2*Hkv_r)*hd` elements per rank; FLOPs `2*T` times that.
- `attention` (fused, FlashAttention-style, compulsory traffic only): decode FLOPs `4*B*C*H*hd/tp`, bytes include KV read `B*C*2*Hkv_r*hd*b_kv` and KV write, where `Hkv_r = max(1, Hkv/tp)` because KV heads replicate when `tp > Hkv`; prefill FLOPs `4*B*(S*(S+1)/2)*H*hd/tp`.
- `o_proj`: weight `H*hd*d/tp`.
- `mlp` (SwiGLU): weight `3*d*d_ff/tp`; FLOPs `6*T*d*d_ff/tp`.
- `moe_router`, `moe_experts`, `moe_shared`: expected distinct experts touched `E_u = E*(1-(1-k/E)^T)` under uniform independent routing (assumption labeled in outputs); expert weight bytes `E_u*3*d*d_ffe*b_w/tp`; FLOPs `6*T*k*d*d_ffe/tp`.
- MLA (DeepSeek-V3 style): weights `q_a`, `q_b`, `kv_a`, `kv_b`, `o`; latent cache `(r_kv + d_rope)*b_kv` bytes per token per layer, replicated per TP rank (no division by `tp`); decode uses weight absorption with attention FLOPs `2*B*C*(H/tp)*(2*r_kv + d_rope)` plus the absorbed `W_UK`/`W_UV` projections, and KV bytes `B*C*(r_kv + d_rope)*b_kv`; prefill uses the non-absorbed form with per-head K and V materialized by `kv_b`.
- `tp_allreduce` (twice per layer when `tp > 1`): ring time `2*(tp-1)/tp * T*d*b_act / (link_gbps*1e9*net_eff) + 2*(tp-1)*link_latency_us`.
- Model-level: `embed` (gather), `final_norm`, `lm_head` (weight `V*d/tp`; logits only for the last token of each sequence in prefill).

Each op carries `kernels`, the kernel-launch count of a standard unfused implementation, used for overhead.

### 8.2 Two-tier memory model

Tier 1 is HBM (`hbm_gb`, `hbm_gbps`).
Tier 2 is an optional capacity tier (`tier2_gb`, `tier2_gbps`, default 0 meaning absent) that stands for next-generation designs such as LPDDR behind a coherent chip-to-chip link or CXL-attached memory.
KV cache fills the HBM budget left after weights first; the remainder spills to tier 2, and a request whose KV exceeds both tiers does not fit.
The attention op reads the HBM-resident KV fraction at HBM bandwidth and the spilled fraction at tier-2 bandwidth concurrently, so its memory time is the maximum of the two tier times.
On-chip caches (L2, shared memory) are not modeled: all compulsory traffic is charged to HBM, which is the standard roofline simplification and is listed as a limitation.
The FlashAttention re-read traffic model from the first plan draft was removed after audit because its L2 fit rule ignored concurrent working sets and could not be validated on available hardware.

### 8.3 Cost and bound class

`t_compute = flops / (peak_tflops[dtype]*1e12*compute_eff)`, `t_memory` as above, `t_overhead = kernels*kernel_overhead_us*1e-6`.
`t_op = max(t_compute, t_memory) / eff + t_overhead` in `additive` overhead mode, where `eff` is the calibrated shape efficiency (`gemm_eff_by_tokens` at the op's matmul row count, or `attn_eff` for attention; 1.0 when uncalibrated).
In `overlap` mode (asynchronous launch queues hide dispatch behind earlier kernels), step time is `max(sum of busy times, sum of overhead times)`.
The mode is a hardware-profile knob; for the local validation it is chosen on the development twin only and frozen before the final run (audit finding I7).
Bound class is the argmax over the busy components (compute, hbm, tier2, network) and overhead; the busy component that sets an op's busy time is its main resource.
Ties are broken deterministically in the order compute, hbm, tier2, network, overhead.

### 8.4 Request metrics

TTFT = prefill time.
TPOT = mean decode-step time over output positions; evaluated exactly when `output <= 64`, otherwise as the mean over 64 evenly spaced positions (midpoint rule; step time is piecewise linear in context, so error is bounded by the curvature at bound transitions; a unit test bounds it against the exact sum).
Decode throughput = `B / TPOT`; end-to-end = `TTFT + output*TPOT`.

### 8.5 Capacity

`weight_bytes_per_rank + B*(S+O)*kv_bytes_per_token_per_rank` must fit `hbm_gb*1e9*capacity_usable + tier2_gb*1e9` (`capacity_usable` default 0.9 absorbs activation workspace), where per-rank KV uses `Hkv_r` for GQA and the full latent for MLA.
`max_batch(S, O)` is solved in closed form and verified by evaluation.

### 8.6 Critical path and elasticity

Execution within a rank is a chain, so the critical path is the op sequence; attribution groups time by (op class, bound).
Elasticity of time with respect to resource `r` is `e_r = -d ln T / d ln r`.
Busy resources are compute, HBM, tier 2 (capacity-tier reads), and network; each is a bound class and an elasticity resource.
In `additive` overhead mode, `T` is a sum of maxima, so away from ties `e_r` equals the share of `T` in which `r` is the largest busy component of an op, and overhead contributes its additive share separately; an op whose overhead exceeds its busy time still attributes its busy time to the busy resource (audit finding I2).
In `overlap` mode each step is `max(sum busy, sum overhead)`, which is all-or-nothing: if overhead is larger, overhead elasticity is 1 and every busy elasticity is 0; otherwise overhead elasticity is 0 and busy shares are taken over the step (audit finding N1).
Claim S2 samples both overhead modes and hardware with and without a capacity tier.
Making the network `f` times faster scales link bandwidth by `f` and link latency by `1/f` together, so its elasticity is the whole network share.
The analytic value is reported and a test checks it against central finite differences at +/- 0.1 percent.
Two objectives are reported and never mixed (audit finding I3): latency elasticity (end-to-end time at fixed batch) and throughput elasticity (output tokens per second at the real-valued capacity-limited batch `B_max`), the latter by finite differences for every resource including capacity.

### 8.7 Intercepts

- Decode knee `B*`: smallest integer batch in `[1, B_cap]` where compute-bound time share is at least 0.5, found by exponential search then bisection (monotone in `B` for fixed context; a property test checks monotonicity).
- Analytic GEMM ridge: `B_ridge = (peak_tflops*1e12*compute_eff) / (hbm_gbps*1e9*mem_eff) * b_w / 2` reported beside `B*` (about 295 for H100 bf16 at efficiency 1).
- Attention-dominance context `C*`: smallest context where attention time share in decode is at least 0.5.
- Capacity wall: `max_batch` versus context curve.
- TP saturation: smallest `tp` in `{1,2,4,8}` beyond which step time stops decreasing by at least 5 percent.
- Any search that does not cross inside its range returns `INCONCLUSIVE` with the range, never a fabricated value.

### 8.8 What-ifs and recommendations

What-ifs apply a multiplicative change to one hardware field (for example `hbm_gbps x1.5`, `hbm_gb x1.5`, `peak x2`, `link_gbps x2`), add a capacity tier (`tier2_gb`, `tier2_gbps`), or change one software field (`kv_dtype fp8`, `weight_dtype fp8`), re-run the archetype at its operating point, and report the latency change at fixed batch and the throughput change at the capacity-limited batch.
The tier-2 intercept is the tier bandwidth at which spilling KV to the capacity tier stops raising capacity-limited throughput.
Recommendations are deterministic text rendered from the ranked what-if table and elasticities; no generative model.

## 9. Repository layout

```text
pyproject.toml            package metadata, extras: plot, measure, dev
README.md                 public overview, results, limitations
BUILD_PLAN.md             this plan
src/llmba/
  __init__.py             version
  specs.py                spec dataclasses, loaders, presets
  presets/models/*.json   public architecture configs with source URLs
  presets/hardware/*.json datasheet specs with source URLs
  presets/twins/*.json    tiny executable twin configs (dev and final)
  ops.py                  operator graph
  cost.py                 cost engine
  analyze.py              phases, request metrics, capacity, elasticities
  sweep.py                grids, dataset, manifest
  intercepts.py           crossovers, what-ifs, recommendations
  report.py               markdown and figures
  measure.py              calibration, twin, validation gate
  cli.py                  command-line interface
studies/                  study JSON files (archetypes, grids)
tests/                    pytest suites
docs/methodology.md       equations, assumptions, limitations
docs/validation.md        validation protocol and recorded results
results/                  generated outputs (git-ignored)
```

Committed: source, presets, studies, tests, docs, selected small evidence tables pasted into docs.
Ignored: `results/`, `.venv/`, caches.

## 10. Toolchain and dependency strategy

Python 3.12 with `uv` for the virtual environment.
Runtime dependencies: none for the analytical engine.
Extras: `plot` (matplotlib), `measure` (torch, used for MPS, CPU, or CUDA timing), `dev` (pytest, ruff).
Versions are recorded in outputs (`platform`, `torch.__version__`, device name); a lock is produced with `uv pip compile` into `requirements-lock.txt` for reproducibility.

## 11. Data acquisition, licensing, and split policy

Model configs are architecture hyperparameters transcribed from public Hugging Face `config.json` files or official papers, cited by URL; no weights are downloaded.
Hardware numbers are transcribed from vendor datasheets, cited by URL; values not found in a primary source are marked `"verified": false` in the preset.
Published throughput reference rows (for S3) are transcribed with URL and retrieval date into `studies/published_reference.json`.

Split policy for empirical validation: calibration uses only standalone microbenchmarks (copy bandwidth counting read plus write bytes, large GEMM, tiny-kernel launch loop, and, after the Milestone 7 replan below, a matmul row-count sweep on generic weight shapes and a KV-streaming decode-attention sweep, all timed as back-to-back loops).
Replan at Milestone 7 (development data only): the first development runs failed P1 because Apple MPS matmul efficiency depends strongly on the row count M (a cliff near M=8) and decode attention attains about a third of copy bandwidth; per-call synchronization also biased sub-millisecond timings.
The structural fix is operator-level calibration: `gemm_eff_by_tokens` (attained over roofline time by M) and `attn_eff`, measured on shapes that are not the twin's; datacenter presets leave both at 1.0.
The overhead mode was chosen as `additive` on development data (MdAPE 0.094 versus 0.134 for `overlap`).
The development twin (`tiny-dev`) and development grid may be used while implementing.
The final twin (`tiny-final`, different width, FFN size, and depth) with its final grid is frozen in `presets/twins/tiny-final.json`, and its SHA-256 recorded in `docs/validation.md`, at the start of Milestone 7 before any twin measurement; the final grid is run once at Milestone 9; any model change after viewing final results invalidates the final claim and requires a new untouched grid.
No efficiency knob may be fitted to twin measurements.

## 12. Testing, fault injection, security, and reproducibility

Test strategy:

- Golden: parameter counts per the S1 rule; KV bytes per token for Llama-3.1-8B equal 131072 at bf16; decode GEMM FLOPs equal `2 * (active_params - input_embedding)` per token within 0.1 percent at small context; decode weight bytes equal active parameters minus embedding and norms exactly.
- Property: step time non-decreasing in batch and context; non-increasing in any hardware bandwidth or peak; memory-bound decode time scales as `1/hbm_gbps` when no KV spills to tier 2.
- Metamorphic: analytic elasticity equals finite difference; doubling every hardware rate halves time when kernel overhead and link latency are zero.
- Fault injection: malformed specs (unknown key, negative size, `n_heads` not divisible by `n_kv_heads`, unknown dtype, tp not dividing heads) must raise `SpecError` naming the field; nearby valid controls must load.
- Determinism: two sweeps produce identical `dataset.csv` SHA-256.
- CLI: subprocess tests for each command and exit codes.
- Measurement: the validation gate is unit tested on synthetic measured and predicted arrays for PASS, FAIL, and INSUFFICIENT_EVIDENCE; the negative control (hardware profile with `hbm_gbps x4`) must FAIL on real measurements.

Security and privacy: no network access at runtime, no services, no credentials.
Trust boundary is user-supplied JSON: validated strictly, size-limited to 1 MB, and never evaluated.
Output directories are created no-clobber (refuse a non-empty target unless `--force`).
No personal paths or machine identifiers beyond device model name are written to evidence.

Reproducibility: outputs record spec hashes, study hash, `llmba_version`, Python version, and for measurements torch version, device, seeds, trial counts, and raw timing arrays; `manifest.json` stores SHA-256 of every artifact.

## 13. Evaluation metrics and decision rules

### Claim S1 (accounting)

Estimand: relative error of computed total and active parameters versus published totals.
Gate: `abs(err) <= 0.01` against exact published counts (safetensors metadata); against rounded published figures (for example "37B activated") the tolerance is 0.01 plus half a unit of the last published digit relative to the figure, because the rounding and the counting convention (embeddings included or not) are not stated; any miss is FAIL.
Disclosure (audit finding N2): the rounded-figure tolerance was adopted after observing DeepSeek-V3 active parameters at +1.49 percent against "37B", which fails the original 1 percent rule; `docs/validation.md` records this, together with the observation that excluding the input embedding gives about 36.6B, inside the rounding interval.

### Claim S2 (elasticity consistency)

Gate: for 200 randomly seeded operating points (seed 0) across presets, analytic and finite-difference elasticities agree within 0.02 absolute at points where no op is within 1 percent of a tie; tied points are excluded and counted.

### Claim S3 (published ceiling)

A separately evaluated prefill-plus-decode time is not a valid upper bound on throughput, because chunked prefill shares weight reads with decode and in-flight batching keeps sequences at mixed progress.
The ceiling is therefore derived without any scheduler assumption (audit finding B1).
Every decode step streams all non-embedding weights `W_read` once (the input embedding is gathered, not streamed) and the KV of every resident sequence; resident KV per step is at most `Cap - W_res`, where `W_res` is all weights including the embedding.
Summing over steps, the number of steps is at least `N*KV_req/(Cap - W_res)`, so per request the HBM traffic is at least `M_req = KV_req * (1 + W_read/(Cap - W_res))` with `KV_req = kv_tok * (O-1) * (S + (O-2)/2)` (the first token comes from prefill).
Then `t_req >= max(F_req / P, M_req / BW)` and aggregate output throughput is at most `O / t_req`.
Every choice raises the ceiling: `Cap = n_gpus * hbm_gb * 2^30` (binary, larger than the decimal datasheet figure), `P` is the dense FP8 peak summed over GPUs and charged for every FLOP, `BW` is summed HBM bandwidth, weights and KV are 1 byte, efficiencies are 1.0, overhead is zero.
Preconditions checked per reference row: dense model, total output tokens per second across all GPUs, no speculative decoding, no KV or weight host offload, no prefix reuse (synthetic random prompts), no sub-8-bit weights or KV.
Estimand: ratio of published measured output throughput to this ceiling (source: TensorRT-LLM performance overview, FP8, transcribed with retrieval date).
Gate: every ratio at most 1.0; a ratio above 1.0 is FAIL and triggers investigation of missing features.
The ratio distribution is reported as achieved fraction of ceiling, with no accuracy claim.

### Claim P1 (empirical closed loop)

Environment: Apple M2 GPU via torch MPS, fp16, recorded device and versions.
Twin size (audit finding B2): the final twin is dense GQA with roughly 0.5 GB of fp16 weights so that predicted memory time per decode step is at least three times predicted overhead at batch 1.
Grid: at least 32 (batch, context) points on a power-of-two batch axis with `batch * context` capped to bound device memory; at least two thirds of the points must have predicted HBM share at least 0.7 under the calibrated profile, checked when the grid is frozen; the same rule applies to the development grid.
Freeze order (audit finding B3): calibrate from microbenchmarks, then freeze and hash `tiny-final` and the final grid, then begin any development twin measurement; every final-grid attempt is logged and the attempt count is reported.
Protocol: warm-up 3 iterations, then 15 timed iterations per grid point with device synchronization at the timing boundary; grid order shuffled with a fixed seed; per-point statistic is the median.
Contention rule: a point whose coefficient of variation exceeds 0.25 is re-measured once; if it still exceeds 0.25 it is invalid.
Support floor: at least 80 percent of grid points valid, otherwise `INSUFFICIENT_EVIDENCE`.
Estimand: median over valid points of `abs(pred - meas)/meas` (MdAPE), with a 95 percent percentile bootstrap interval over grid points (2000 resamples, seed 0).
Knee rule (one definition for both curves, audit finding I1): at the shortest context, the knee is the smallest grid batch whose step time is at least 1.5 times the batch-1 step time; `NONE` if no batch qualifies.
Proposed threshold (a decision criterion, not an external requirement): PASS if the bootstrap upper bound of MdAPE is at most 0.30 and the predicted knee is within one power-of-two grid step of the measured knee (both `NONE` also agrees).
Rationale: roofline models are first-order estimates, and 30 percent separates a structurally correct model from a wrong-regime model.
Negative control: the same measurements graded on MdAPE alone against a profile with `hbm_gbps x4` must FAIL with a bootstrap lower bound above 0.30; this is discriminating because most final-grid points are predicted memory-bound, so a 4x bandwidth error moves their predictions by far more than 30 percent.
A structural control (attention KV traffic removed) is reported without a gate.
Revisit trigger: if the development grid shows systematic error from a structural mismatch (for example kernel count or overhead mode), fix the structure on development data only, never on the final grid.

## 14. User experience

CLI commands: `presets`, `analyze`, `sweep`, `intercepts`, `recommend`, `report`, `measure calibrate`, `measure validate`, `check-published`.
Every command supports `--json` for machine output where it prints tables.
Exit codes: 0 success, 1 gate FAIL or INCONCLUSIVE where a verdict is requested, 2 usage or spec error, 3 BLOCKED (device unavailable).
Figures: hierarchical roofline with ops plotted, decode step time versus batch per hardware with knees marked, intercept map (knee batch versus context), elasticity bars per archetype, capacity wall.

## 15. Milestones, acceptance gates, and checkpoint boundaries

The owner asked for local work without commits; each milestone ends at a checkpoint boundary where a focused commit would be made later.

### Milestone 0: feasibility

Deliverables: `pyproject.toml`, venv, `specs.py` with Llama-3.1-8B and H100 presets, minimal `ops.py` and `cost.py` for dense decode, one golden test, MPS spike results recorded in `docs/validation.md`.
Acceptance gate: Llama-3.1-8B parameter count within 1 percent; predicted batch-1 decode step on H100 is weight-bandwidth-bound; the MPS spike (a throwaway chained-GEMM configuration, not `tiny-dev` or `tiny-final`) shows batch-1 GEMV effective bandwidth within 30 percent of measured copy bandwidth and a step time at batch 128 at least 1.5 times the batch-1 time (a measurable knee).
Kill or fallback: if MPS timing is unusable, fall back to CPU timing with Accelerate numpy and narrow P1 to CPU.

### Milestone 1: specs and presets

Deliverables: all model presets (Llama-3.1-8B, Llama-3.1-70B, Llama-2-7B, Mixtral-8x7B, DeepSeek-V3) and hardware presets (A100-80GB, H100-SXM, H200, B200, L4, MI300X) with source URLs; strict validation; fault-injection tests.
Acceptance gate: S1 passes; every fault case raises the right field error; every control loads.

### Milestone 2: operator graph and cost engine

Deliverables: `ops.py`, `cost.py`, `analyze.py` with GQA, MoE, MLA, TP, memory hierarchy, request metrics, capacity, elasticities.
Acceptance gate: golden, property, and metamorphic tests pass, including S2 and the TPOT sampling bound.

### Milestone 3: CLI analyze and critical path

Deliverables: `cli.py` with `presets` and `analyze`; human table and `--json`.
Acceptance gate: subprocess test of the documented Workflow A; output includes phase times, top critical-path ops, bound classes, capacity headroom, and elasticities.

### Milestone 4: sweeps and dataset synthesis

Deliverables: `sweep.py`, `sweep` command, manifest, no-clobber output.
Acceptance gate: determinism test passes; manifest hashes verify; refusing to overwrite is tested.

### Milestone 5: intercepts and recommendations

Deliverables: `intercepts.py`, archetypes in `studies/`, `intercepts` and `recommend` commands.
Acceptance gate: knee monotonicity property; INCONCLUSIVE path tested; B* agrees with analytic ridge within a factor of 2 for dense models at short context; recommendations cite numbers present in the table.

### Milestone 6: report and figures (portfolio-ready core checkpoint)

Deliverables: `report.py`, `report` command, `studies/next_gen_memory.json`, generated report.
Acceptance gate: report generates from a clean venv with the `plot` extra; figures exist and are non-empty; report numbers equal `results.json` numbers.

### Milestone 7: local calibration and development validation

Deliverables: `measure.py`, calibration, then frozen and hashed `presets/twins/tiny-final.json` (before any twin measurement), then `presets/twins/tiny-dev.json`, `measure calibrate` and `measure validate`.
Acceptance gate: gate logic unit tests pass; the frozen final grid meets the two-thirds memory-share rule; development validation runs end to end on MPS and records a verdict; the negative control FAILs on development data; the overhead mode is chosen on development data and frozen.

### Milestone 8: published ceiling check

Deliverables: `studies/published_reference.json`, `check-published` command.
Acceptance gate: S3 evaluated and recorded; if no public rows can be verified, record BLOCKED with the reason and keep the claim out of the README.

### Milestone 9: final holdout, documentation, and release acceptance

Deliverables: one final validation run of the grid frozen at Milestone 7 (hash verified before running), `README.md`, `docs/methodology.md`, `docs/validation.md`.
Acceptance gate: final run recorded with its verdict (PASS, FAIL, or INSUFFICIENT_EVIDENCE reported truthfully); clean-checkout acceptance passes.

## 16. Hardware and compute matrix

| Workload | Preferred resource | Minimum resource | Expected duration | Checkpoint strategy | Required evidence | Fallback |
| --- | --- | --- | --- | --- | --- | --- |
| Analytical engine, sweeps, report | Local Apple M2 | Any CPU with Python 3.12 | Seconds to minutes | Deterministic re-run | Tests, manifests | None needed |
| Calibration and twin validation | Local Apple M2 GPU (MPS) | Local M2 CPU | Minutes | Per-point raw timings saved as they complete | Profile JSON, validation JSON | CPU timing with narrowed claim |
| CUDA twin qualification (follow-on F1) | Colab Pro GPU | Colab T4 | Under one hour | Same evidence format, no-clobber run id | Same validation JSON on CUDA | Remains follow-on |

Capability probe: `llmba measure calibrate` detects `cuda`, then `mps`, then `cpu`, records the device, and returns exit code 3 (`BLOCKED`) when the requested device is unavailable.
No paid compute is required for V1.

## 17. Documentation and interview walkthrough

README: problem, quickstart, the five workflows, headline figures, validation results table, limitations, prior art, reproduction commands.
`docs/methodology.md`: every equation in section 8 with assumptions.
`docs/validation.md`: protocol, spike evidence, development and final results with hashes.
Walkthrough: roofline figure, MoE knee shift, elasticity flip for long context, validation table, negative control.

## 18. Risks, kill gates, and fallback claim tree

| Risk | Mitigation | Fallback |
| --- | --- | --- |
| Datasheet values inconsistent across sources | Cite primary URL, flag unverified | Mark preset `verified: false` in reports |
| MPS timing noise on an 8 GB shared machine | Medians, CV invalidation, support floor | INSUFFICIENT_EVIDENCE reported truthfully |
| Structural mismatch between twin kernels and op graph | `kernels` count per op, fused torch calls, overhead mode chosen on dev data | Report per-regime error and narrow P1 |
| Uniform-routing MoE assumption unrealistic | Labeled assumption | State that skewed (correlated) routing touches fewer experts and anti-correlated routing can touch more; no bound claimed |
| Published throughput rows unverifiable | Transcribe only verified rows | S3 BLOCKED, omitted from README claims |

Fallback claim tree for P1, evaluated from the same final run: (1) full P1 PASS; (2) MdAPE gate passes but knee misses, report knee as FAIL; (3) MdAPE fails, report per-regime error (memory-bound versus compute-bound points) and the failing regime; (4) INSUFFICIENT_EVIDENCE.

Limitations to publish: analytical first-order model; no cycle accuracy; no on-chip cache model; uniform independent expert routing; datacenter GPU outputs are unvalidated model predictions; Apple M2 uses LPDDR unified memory, not HBM.

## 19. Final release and clean-checkout acceptance

From a fresh copy of the source tree (excluding `results/`, `.venv/`): create a venv with `uv`, install `.[dev,plot,measure]`, run `ruff check`, run `pytest`, run Workflow A, B, C, D, and `measure validate` against the recorded calibration.
Release acceptance requires all tests passing, all documented commands succeeding, and README numbers matching generated outputs.
No tag, push, or publication is performed without owner authorization.

## 20. Authoritative sources

- Hugging Face model configs for Llama-3.1-8B, Llama-3.1-70B, Llama-2-7B, Mixtral-8x7B-v0.1, DeepSeek-V3 (URLs recorded in each preset).
- NVIDIA A100, H100, H200, B200 (HGX and DGX B200), L4 product pages and briefs, NVIDIA architecture blogs and tuning guides, and the AMD MI300X datasheet (URLs recorded in each preset; values not stated by a primary source are marked `verified: false`).
- NVIDIA TensorRT-LLM performance overview, `https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html` (measured FP8 reference rows).
- Williams, Waterman, Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures", CACM 2009.
- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", NeurIPS 2022.
- DeepSeek-AI, "DeepSeek-V3 Technical Report", 2024.
- Patarasuk and Yuan, "Bandwidth optimal all-reduce algorithms for clusters of workstations", JPDC 2009.
- PyTorch MPS backend documentation.
