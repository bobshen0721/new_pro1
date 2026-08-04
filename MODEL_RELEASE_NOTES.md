# Offline public models (`models-v1`)

This release contains split, checksum-verified offline copies of three publicly downloadable models used by the application:

- `Systran/faster-whisper-large-v2`
- `JacobLinCool/TEA-ASR-1.1`
- `Qwen/Qwen3-ForcedAligner-0.6B`

Each asset is smaller than 2 GiB. Download `restore_release_models.ps1` and `model-release-manifest.json`, then run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\restore_release_models.ps1 -ReleaseTag models-v1
```

The restore script verifies every part and the reassembled archive with SHA-256 before extraction.

`pyannote/speaker-diarization-community-1` is intentionally excluded. It is gated upstream, so each user must accept its Hugging Face conditions and download it with their own read-only HF Token. See `MODEL_RELEASE.md` for source and license details.
