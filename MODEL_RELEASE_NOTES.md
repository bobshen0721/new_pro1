# TEA-ASR-1.1-mini offline model (`models-v2`)

This release contains a checksum-verified offline copy of only the newly selected model:

- `JacobLinCool/TEA-ASR-1.1-mini`

Whisper large-v2 and Qwen3-ForcedAligner-0.6B already exist in `models-v1`, so they are intentionally not duplicated here. Download `restore_release_models.ps1` and `model-release-manifest.json`, then run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\restore_release_models.ps1 -ReleaseTag models-v2
```

The restore script verifies the mini part and reassembled archive with SHA-256 before extraction.

`pyannote/speaker-diarization-community-1` remains excluded because each user must accept its Hugging Face conditions and download it with their own read-only HF Token. See `MODEL_RELEASE.md` for the source revision and license details.
