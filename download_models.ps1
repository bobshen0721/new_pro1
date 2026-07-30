param([string]$ModelRoot = (Join-Path $PSScriptRoot "models"))
$ErrorActionPreference = "Stop"

if (-not $env:HF_TOKEN) {
  Write-Host "請先設定新的 HF_TOKEN：" -ForegroundColor Yellow
  Write-Host '$env:HF_TOKEN="hf_你的新權杖"'
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
    ("Systran/faster-whisper-large-v3", "faster-whisper-large-v3"),
    ("pyannote/speaker-diarization-community-1", "pyannote-community-1"),
]:
    target = root / folder
    print(f"下載 {repo_id} -> {target}")
    snapshot_download(repo_id=repo_id, local_dir=str(target), token=token)

print("完成。請把整個 models 資料夾帶進公司環境。")
'@ | python -

if ($LASTEXITCODE -ne 0) {
  throw "模型下載失敗。請確認已接受 pyannote 模型條款。"
}
