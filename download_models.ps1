param([string]$ModelRoot = (Join-Path $PSScriptRoot "models"))
$ErrorActionPreference = "Stop"

if (-not $env:HF_TOKEN) {
  Write-Host "Set a read-only HF_TOKEN first:" -ForegroundColor Yellow
  Write-Host '$env:HF_TOKEN="hf_your_read_token"'
  exit 1
}

New-Item -ItemType Directory -Force -Path $ModelRoot | Out-Null
python -m pip install --upgrade huggingface-hub
$env:MODEL_ROOT = $ModelRoot

@'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(os.environ["MODEL_ROOT"]).resolve()
token = os.environ["HF_TOKEN"]

for repo_id, folder in [
    ("Systran/faster-whisper-large-v2", "faster-whisper-large-v2"),
    ("JacobLinCool/TEA-ASR-1.1", "TEA-ASR-1.1"),
    ("Qwen/Qwen3-ForcedAligner-0.6B", "Qwen3-ForcedAligner-0.6B"),
    ("pyannote/speaker-diarization-community-1", "pyannote-community-1"),
]:
    target = root / folder
    print(f"Downloading {repo_id} -> {target}")
    snapshot_download(repo_id=repo_id, local_dir=str(target), token=token)

print("Done. Copy the complete models directory to the offline environment.")
'@ | python -

if ($LASTEXITCODE -ne 0) {
  throw "Model download failed. Confirm that the pyannote model conditions were accepted."
}
