import json

import pytest

from llmba.ops import param_counts
from llmba.specs import (
    SpecError,
    Workload,
    load_hardware,
    load_model,
    preset_names,
    with_overrides,
)

MODELS = preset_names("models")


def _rounding_tol(value: float) -> float:
    """1% plus half a unit of the last published digit (claim S1)."""
    text = repr(value).rstrip("0").rstrip(".")
    decimals = len(text.split(".")[1]) if "." in text else 0
    return 0.01 + 0.5 * 10**-decimals / value


@pytest.mark.parametrize("name", MODELS)
def test_param_counts_match_published(name):
    m = load_model(name)
    total, active = param_counts(m)
    assert abs(total / 1e9 / m.published_params_b - 1) <= _rounding_tol(m.published_params_b)
    assert abs(active / 1e9 / m.published_active_params_b - 1) <= _rounding_tol(
        m.published_active_params_b)


def test_exact_published_counts_within_one_percent():
    for name in ("llama-3.1-8b", "llama-3.1-70b", "llama-2-7b"):
        total, _ = param_counts(load_model(name))
        assert abs(total / 1e9 / load_model(name).published_params_b - 1) <= 0.01


@pytest.mark.parametrize("name", preset_names("hardware"))
def test_hardware_presets_load_with_sources(name):
    hw = load_hardware(name)
    assert hw.source_urls and hw.peak("bf16") > 0


def _write(tmp_path, data):
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(data))
    return str(p)


BASE = json.loads(json.dumps(load_model("llama-3.1-8b").__dict__, default=list))


@pytest.mark.parametrize("mutation, field", [
    ({"n_layerz": 3}, "n_layerz"),
    ({"d_model": -1}, "d_model"),
    ({"n_kv_heads": 5}, "n_kv_heads"),
    ({"moe": {"n_experts": 4, "top_k": 8, "d_ff_expert": 10}}, "moe.top_k"),
    ({"moe": {"n_experts": 4, "top_k": 2, "d_ff_expert": 10, "bogus": 1}}, "moe.bogus"),
    ({"schema_version": 2}, "schema_version"),
])
def test_malformed_model_names_field(tmp_path, mutation, field):
    with pytest.raises(SpecError, match=field.replace(".", r"\.")):
        load_model(_write(tmp_path, {**BASE, **mutation}))


def test_valid_control_loads(tmp_path):
    ok = {**BASE, "moe": {"n_experts": 4, "top_k": 2, "d_ff_expert": 10}}
    assert load_model(_write(tmp_path, ok)).moe.top_k == 2


def test_invalid_json_and_unknown_preset(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{nope")
    with pytest.raises(SpecError, match="invalid JSON"):
        load_model(str(p))
    with pytest.raises(SpecError, match="unknown preset"):
        load_model("gpt-9")


def test_oversized_spec_rejected(tmp_path):
    p = tmp_path / "big.json"
    p.write_text(" " * 1_000_001)
    with pytest.raises(SpecError, match="exceeds"):
        load_model(str(p))


def test_workload_validation():
    with pytest.raises(SpecError, match="kv_dtype"):
        Workload(kv_dtype="int3")
    with pytest.raises(SpecError, match="batch"):
        Workload(batch=0)
    with pytest.raises(SpecError, match="tp"):
        Workload(tp=3).check(load_model("llama-3.1-8b"))
    Workload(tp=16).check(load_model("llama-3.1-70b"))  # kv heads replicate: control passes


def test_hardware_overrides_and_missing_dtype():
    hw = with_overrides(load_hardware("a100-80gb"), {"mem_eff": "0.5", "peak_tflops.fp8": 1})
    assert hw.mem_eff == 0.5 and hw.peak("fp8") == 1e12
    with pytest.raises(SpecError, match="fp8"):
        load_hardware("a100-80gb").peak("fp8")
    with pytest.raises(SpecError, match="mem_eff"):
        with_overrides(hw, {"mem_eff": 1.5})
    with pytest.raises(SpecError, match="name"):
        with_overrides(hw, {"name": 1})
