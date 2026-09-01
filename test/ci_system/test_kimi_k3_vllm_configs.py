import json
import os
import subprocess
from pathlib import Path

CONFIGS = {
    "attn_tp8_moe_tp8.sh": {"tensor_parallel": "8", "data_parallel": None},
    "attn_tp8_moe_ep8.sh": {"tensor_parallel": "8", "data_parallel": None},
    "attn_dp8_moe_ep8.sh": {"tensor_parallel": "1", "data_parallel": "8"},
}


def _last_value(args, option):
    index = len(args) - 1 - args[::-1].index(option)
    return args[index + 1]


def _run_config(config, mock_vllm, extra_args=(), **env_overrides):
    env = os.environ.copy()
    for name in ("MODEL_DIR", "DRAFT_DIR", "HTTP_PORT", "VLLM_HOST"):
        env.pop(name, None)
    env.update(VLLM_REAL_BIN=str(mock_vllm), **env_overrides)
    result = subprocess.run(
        [config, *extra_args],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=True,
    )
    return result.stdout.splitlines()


def test_vllm_configs_accept_checkpoint_and_slurm_overrides(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = (
        repo_root / "test" / "agentic_benchmark" / "kimi_k3" / "vllm" / "configs"
    )
    mock_vllm = tmp_path / "vllm"
    mock_vllm.write_text("#!/usr/bin/bash\nprintf '%s\\n' \"$@\"\n")
    mock_vllm.chmod(0o755)

    node_args = (
        "--distributed-executor-backend",
        "mp",
        "--nnodes",
        "2",
        "--node-rank",
        "1",
        "--master-addr",
        "node0",
        "--master-port",
        "29500",
        "--host",
        "0.0.0.0",
        "--served-model-name",
        "nvidia/Kimi-K3-NVFP4",
        "--headless",
    )

    for name, expected in CONFIGS.items():
        args = _run_config(
            config_dir / name,
            mock_vllm,
            node_args,
            MODEL_DIR="/models/Kimi-K3-NVFP4",
            DRAFT_DIR="/models/Kimi-K3-DSpark",
            HTTP_PORT="9002",
        )
        assert args[0] == "serve"
        assert _last_value(args, "--model") == "/models/Kimi-K3-NVFP4"
        assert (
            _last_value(args, "--tensor-parallel-size") == expected["tensor_parallel"]
        )
        assert _last_value(args, "--host") == "0.0.0.0"
        assert _last_value(args, "--port") == "9002"
        assert _last_value(args, "--nnodes") == "2"
        assert _last_value(args, "--node-rank") == "1"
        assert "--headless" in args
        speculative = json.loads(_last_value(args, "--speculative-config"))
        assert speculative["model"] == "/models/Kimi-K3-DSpark"
        if expected["data_parallel"] is None:
            assert "--data-parallel-size" not in args
        else:
            assert (
                _last_value(args, "--data-parallel-size") == expected["data_parallel"]
            )


def test_vllm_config_keeps_local_defaults(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    config = (
        repo_root
        / "test"
        / "agentic_benchmark"
        / "kimi_k3"
        / "vllm"
        / "configs"
        / "attn_tp8_moe_tp8.sh"
    )
    mock_vllm = tmp_path / "vllm"
    mock_vllm.write_text("#!/usr/bin/bash\nprintf '%s\\n' \"$@\"\n")
    mock_vllm.chmod(0o755)

    args = _run_config(config, mock_vllm)
    assert _last_value(args, "--model") == "nvidia/Kimi-K3-NVFP4"
    assert _last_value(args, "--host") == "127.0.0.1"
    assert _last_value(args, "--port") == "8002"
    speculative = json.loads(_last_value(args, "--speculative-config"))
    assert speculative["model"] == "Inferact/Kimi-K3-DSpark"


def test_kimi_k3_pareto_scripts_keep_diagnostics_opt_in():
    repo_root = Path(__file__).resolve().parents[2]
    bench_root = repo_root / "test" / "agentic_benchmark" / "kimi_k3"

    vllm_sbatch = (bench_root / "vllm" / "run_pareto.sbatch").read_text()
    assert "KIMI_K3_PREFIX_CACHE_RETENTION_INTERVAL=${" in vllm_sbatch
    assert 'if [[ -n "$PREFIX_CACHE_RETENTION_INTERVAL" ]]' in vllm_sbatch
    assert "--prefix-cache-retention-interval" in vllm_sbatch
    assert "metrics-before-client.prom" in vllm_sbatch
    assert "metrics-after-client.prom" in vllm_sbatch

    tokenspeed_sbatch = (
        bench_root / "tokenspeed" / "run_pareto.sbatch"
    ).read_text()
    assert "KIMI_K3_BUILD_RUNTIME=${KIMI_K3_BUILD_RUNTIME:-0}" in tokenspeed_sbatch
    assert "stale_cutlass_metadata=(" in tokenspeed_sbatch
    assert 'v = m.version("nvidia-cutlass-dsl")' in tokenspeed_sbatch
    assert "built-runtime.sha256" in tokenspeed_sbatch
