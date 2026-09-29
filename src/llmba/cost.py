"""Roofline cost of each op: time, per-resource components, and the binding resource."""

from __future__ import annotations

from dataclasses import dataclass

from .ops import Op
from .specs import HardwareSpec, SpecError

COMPUTE, HBM, TIER2, NETWORK, OVERHEAD = "compute", "hbm", "tier2", "network", "overhead"
BUSY = (COMPUTE, HBM, TIER2, NETWORK)  # resources that overlap within an op; tie-break order
BOUNDS = (*BUSY, OVERHEAD)


@dataclass(frozen=True)
class OpCost:
    op: Op
    busy: dict[str, float]  # seconds per occurrence on each overlapping resource
    t_overhead: float

    @property
    def main(self) -> str:
        """Busy resource that sets the op's busy time."""
        return max(BUSY, key=lambda b: (self.busy[b], -BUSY.index(b)))

    @property
    def t_busy(self) -> float:
        return self.busy[self.main]

    @property
    def bound(self) -> str:
        return OVERHEAD if self.t_overhead > self.t_busy else self.main

    @property
    def time(self) -> float:
        """Seconds for all `count` occurrences (additive overhead)."""
        return self.op.count * (self.t_busy + self.t_overhead)


def cost_op(op: Op, hw: HardwareSpec, compute_dtype: str) -> OpCost:
    busy = dict.fromkeys(BUSY, 0.0)
    if op.flops:
        busy[COMPUTE] = op.flops / (hw.peak(compute_dtype) * hw.compute_eff)
    busy[HBM] = op.hbm_bytes / (hw.hbm_gbps * 1e9 * hw.mem_eff)
    if op.tier2_bytes:
        busy[TIER2] = op.tier2_bytes / (hw.tier2_gbps * 1e9 * hw.mem_eff)
    if op.kind == "comm":
        p = op.ranks
        if hw.link_gbps <= 0:
            raise SpecError(f"link_gbps: {hw.name} has no interconnect for tp={p}")
        # Ring all-reduce: reduce-scatter + all-gather, 2(p-1) latency-bound steps.
        busy[NETWORK] = 2 * (p - 1) / p * op.comm_bytes / (hw.link_gbps * 1e9 * hw.net_eff)
        busy[NETWORK] += 2 * (p - 1) * hw.link_latency_us * 1e-6
    eff = hw.attn_eff if op.kind == "attention" else hw.gemm_eff(op.tokens)
    if eff != 1.0:
        busy = {k: v / eff for k, v in busy.items()}
    return OpCost(op, busy, op.kernels * hw.kernel_overhead_us * 1e-6)


def cost_ops(ops: list[Op], hw: HardwareSpec, compute_dtype: str) -> list[OpCost]:
    return [cost_op(op, hw, compute_dtype) for op in ops]


def time_by_bound(costs: list[OpCost], overhead_mode: str = "additive") -> dict[str, float]:
    """Step time attributed to resources. Additive: each op's busy time goes to its main resource
    and launch overhead adds on top. Overlap: step time is max(total busy, total overhead), so
    only the larger side is on the critical path."""
    out = dict.fromkeys(BOUNDS, 0.0)
    for c in costs:
        out[c.main] += c.op.count * c.t_busy
        out[OVERHEAD] += c.op.count * c.t_overhead
    if overhead_mode == "overlap":
        busy = sum(out[b] for b in BUSY)
        if busy >= out[OVERHEAD]:
            out[OVERHEAD] = 0.0
        else:
            out = {**dict.fromkeys(BUSY, 0.0), OVERHEAD: out[OVERHEAD]}
    return out
