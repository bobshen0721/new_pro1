[CmdletBinding()]
param(
  [string]$ModelRoot = (Join-Path $PSScriptRoot "models"),
  [switch]$VerifyOnly
)

$ErrorActionPreference = "Stop"
$modelRootPath = [IO.Path]::GetFullPath($ModelRoot)
New-Item -ItemType Directory -Force -Path $modelRootPath | Out-Null

if (-not $VerifyOnly) {
  if (-not $env:HF_TOKEN) {
    Write-Host "Set a read-only HF_TOKEN first:" -ForegroundColor Yellow
    Write-Host '$env:HF_TOKEN="hf_your_read_token"'
    exit 1
  }

  python -m pip install --upgrade "huggingface-hub>=0.34,<2"
  if ($LASTEXITCODE -ne 0) {
    throw "Unable to install huggingface-hub."
  }

  $env:HF_HUB_OFFLINE = "0"
  $env:TRANSFORMERS_OFFLINE = "0"
}

$env:MODEL_ROOT = $modelRootPath
$env:VERIFY_ONLY = if ($VerifyOnly) { "1" } else { "0" }

@'
import os
from pathlib import Path

root = Path(os.environ["MODEL_ROOT"]).resolve()
verify_only = os.environ.get("VERIFY_ONLY") == "1"
token = os.environ.get("HF_TOKEN")

models = [
    {
        "repo_id": "Systran/faster-whisper-large-v2",
        "revision": "f0fe81560cb8b68660e564f55dd99207059c092e",
        "folder": "faster-whisper-large-v2",
        "required_files": ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"),
    },
    {
        "repo_id": "JacobLinCool/TEA-ASR-1.1-mini",
        "revision": "98c58048572b44839dfcfa60de3ad7e365a5b232",
        "folder": "TEA-ASR-1.1-mini",
        "required_files": (
            "added_tokens.json",
            "chat_template.jinja",
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ),
    },
    {
        "repo_id": "Qwen/Qwen3-ForcedAligner-0.6B",
        "revision": "c7cbfc2048c462b0d63a45797104fc9db3ad62b7",
        "folder": "Qwen3-ForcedAligner-0.6B",
        "required_files": (
            "config.json",
            "merges.txt",
            "model.safetensors",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "vocab.json",
        ),
    },
    {
        "repo_id": "pyannote/speaker-diarization-community-1",
        "revision": "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee",
        "folder": "pyannote-community-1",
        "required_files": (
            "config.yaml",
            "embedding/pytorch_model.bin",
            "plda/plda.npz",
            "plda/xvec_transform.npz",
            "segmentation/pytorch_model.bin",
        ),
    },
]


def verify_model(spec: dict[str, object]) -> None:
    target = root / str(spec["folder"])
    missing = [
        name
        for name in spec["required_files"]
        if not (target / str(name)).is_file() or (target / str(name)).stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError(
            f"Incomplete model directory: {target}; missing or empty: {', '.join(missing)}"
        )
    print(f"Verified {spec['repo_id']} -> {target}")


if not verify_only:
    from huggingface_hub import snapshot_download

    for spec in models:
        target = root / str(spec["folder"])
        print(f"Downloading {spec['repo_id']} @ {spec['revision']} -> {target}")
        snapshot_download(
            repo_id=str(spec["repo_id"]),
            revision=str(spec["revision"]),
            local_dir=str(target),
            token=token,
        )
        verify_model(spec)
else:
    for spec in models:
        verify_model(spec)

legacy_tea = root / "TEA-ASR-1.1"
if legacy_tea.is_dir():
    print(
        "Note: legacy models/TEA-ASR-1.1 is not used by the current app. "
        "Do not rename it; the mini checkpoint must be downloaded separately."
    )

print(f"All four offline model directories are ready under: {root}")
'@ | python -

if ($LASTEXITCODE -ne 0) {
  if ($VerifyOnly) {
    throw "Offline model verification failed. Recopy the complete models directory."
  }
  throw "Model download failed. Confirm network access, disk space, HF_TOKEN, and accepted pyannote conditions."
}
