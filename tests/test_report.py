import json
from pathlib import Path

import pytest

from llmba.report import build_report
from llmba.sweep import verify_manifest

STUDY = json.loads((Path(__file__).parents[1] / "studies/next_gen_memory.json").read_text())
SMALL = {**STUDY,
         "archetypes": STUDY["archetypes"][:2],
         "focus": STUDY["focus"][:1],
         "scaling": {**STUDY["scaling"], "hardware": ["h100-sxm", "h200"], "batch": [1, 16, 256]},
         "intercept_map": {**STUDY["intercept_map"], "context": [1024, 8192],
                           "models": STUDY["intercept_map"]["models"][:2]},
         "tier2": {"gb": 480, "gbps": [64, 900]},
         "sweep": {"models": ["llama-3.1-8b"], "hardware": ["h100-sxm"], "batch": [1, 64],
                   "prompt_len": [1024], "output_len": [128]}}


@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    pytest.importorskip("matplotlib")
    base = tmp_path_factory.mktemp("r")
    return build_report(SMALL, base / "a"), build_report(SMALL, base / "b")


def test_report_files_and_manifest(reports):
    a, _ = reports
    for name in ("report.md", "results.json", "dataset.csv", "roofline.png", "decode_scaling.png",
                 "intercept_map.png", "elasticity.png", "tier2.png", "published.png"):
        assert (a / name).stat().st_size > 500, name
    assert verify_manifest(a) == []


def test_report_numbers_come_from_results(reports):
    a, _ = reports
    res = json.loads((a / "results.json").read_text())
    md = (a / "report.md").read_text()
    p = res["published"]
    assert f"median {p['median_ratio']:.2f}" in md and p["verdict"] in md
    for row in res["intercept_map"]["models"]:
        assert f"| {row['unconstrained_gemm_knee']} |" in md
    for r in res["recommendations"]:
        assert f"| {r['max_tokens_per_s']:.0f} |" in md


def test_results_and_dataset_byte_reproducible(reports):
    a, b = reports
    for name in ("results.json", "dataset.csv", "report.md"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
