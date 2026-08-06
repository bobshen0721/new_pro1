# Offline public models (`models-v2`)

This release contains split, checksum-verified offline copies of the three publicly downloadable models used by the current application:

- `Systran/faster-whisper-large-v2`
- `JacobLinCool/TEA-ASR-1.1-mini`
- `Qwen/Qwen3-ForcedAligner-0.6B`

Each asset is smaller than 2 GiB. Download `restore_release_models.ps1` and `model-release-manifest.json`, then run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\restore_release_models.ps1 -ReleaseTag models-v2
```

The restore script verifies every part and the reassembled archive with SHA-256 before extraction.

`pyannote/speaker-diarization-community-1` is intentionally excluded. It is gated upstream, so each user must accept its Hugging Face conditions and download it with their own read-only HF Token. See `MODEL_RELEASE.md` for source revisions and license details.

`models-v1` remains available for the previous 2B `TEA-ASR-1.1` application version.
