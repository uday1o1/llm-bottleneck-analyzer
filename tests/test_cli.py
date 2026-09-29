import json
import subprocess
import sys

import pytest

from llmba.cli import main


def run(*args):
    return subprocess.run([sys.executable, "-m", "llmba.cli", *args], capture_output=True,
                          text=True, timeout=600)


def test_workflow_a_human_and_json():
    r = run("analyze", "--model", "llama-3.1-70b", "--hw", "h100-sxm", "--tp", "4", "--batch",
            "64", "--prompt", "2048", "--output", "512")
    assert r.returncode == 0, r.stderr
    for text in ("TTFT", "TPOT", "decode: time share", "fits=True", "attn_allreduce", "ridge"):
        assert text in r.stdout
    j = json.loads(run("analyze", "--model", "llama-3.1-8b", "--hw", "h100-sxm", "--json").stdout)
    assert j["capacity"]["fits"] and j["critical_path"]["decode"][0]["op"] == "mlp"


def test_usage_and_spec_errors_exit_2(capsys):
    assert main(["analyze", "--model", "nope", "--hw", "h100-sxm"]) == 2
    assert main(["analyze", "--model", "llama-3.1-8b", "--hw", "a100-80gb",
                 "--act-dtype", "fp8"]) == 2
    assert main(["analyze", "--model", "llama-3.1-8b", "--hw", "h100-sxm", "--tp", "3"]) == 2
    assert main(["analyze", "--model", "llama-3.1-8b", "--hw", "h100-sxm", "--set", "bogus"]) == 2
    with pytest.raises(SystemExit) as e:
        main(["analyze"])
    assert e.value.code == 2


def test_overrides_change_result():
    base = json.loads(run("analyze", "--model", "llama-3.1-8b", "--hw", "h100-sxm",
                          "--json").stdout)
    slow = json.loads(run("analyze", "--model", "llama-3.1-8b", "--hw", "h100-sxm", "--json",
                          "--set", "hbm_gbps=1000").stdout)
    assert slow["metrics"]["tpot_s"] > 2 * base["metrics"]["tpot_s"]


def test_workflow_b_c_and_published(tmp_path):
    r = run("sweep", "--models", "llama-3.1-8b,mixtral-8x7b", "--hw", "h100-sxm,h200",
            "--batch", "1,64", "--out", str(tmp_path / "s"))
    assert r.returncode == 0 and "wrote 8 rows" in r.stdout, r.stderr
    assert run("sweep", "--out", str(tmp_path / "s")).returncode == 2  # no clobber
    r = run("intercepts", "--model", "deepseek-v3", "--hw", "h200", "--tp", "8",
            "--weight-dtype", "fp8")
    assert r.returncode == 0 and "capacity wall" in r.stdout, r.stderr
    r = run("recommend", "--model", "llama-3.1-8b", "--hw", "h100-sxm", "--batch", "16")
    assert r.returncode == 0 and "Capacity tier" in r.stdout, r.stderr
    r = run("check-published")
    assert r.returncode == 0 and "PASS" in r.stdout, r.stderr


def test_bad_inputs_exit_2_not_traceback():
    assert main(["analyze", "--model", "llama-3.1-8b", "--hw", "h100-sxm",
                 "--set", "hbm_gbps=4800.5"]) == 0
    assert main(["analyze", "--model", "llama-3.1-8b", "--hw", "h100-sxm",
                 "--set", "hbm_gbps=abc"]) == 2
    assert main(["recommend"]) == 2
    assert main(["measure", "validate", "--profile", "h100-sxm", "--twin", "tiny-dev",
                 "--out", "/nonexistent-never-created"]) == 2
    assert main(["measure", "validate", "--profile", "h100-sxm", "--twin", "tiny-dev",
                 "--out", "/x", "--set", "foo"]) == 2
