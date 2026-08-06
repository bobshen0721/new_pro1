# 模型 Release 說明

GitHub Release `models-v2` 提供現行應用程式使用的三個公開模型，並以未壓縮 `tar` 封裝後切成小於 2 GiB 的分片。`restore_release_models.ps1` 會逐片驗證 SHA-256、重組封裝、再次驗證完整封裝，最後解壓到本機 `models` 資料夾。

## 收錄模型

| 本機資料夾 | 原始來源 | 固定 revision | 授權 |
|---|---|---|---|
| `faster-whisper-large-v2` | [Systran/faster-whisper-large-v2](https://huggingface.co/Systran/faster-whisper-large-v2) | `f0fe81560cb8b68660e564f55dd99207059c092e` | MIT |
| `TEA-ASR-1.1-mini` | [JacobLinCool/TEA-ASR-1.1-mini](https://huggingface.co/JacobLinCool/TEA-ASR-1.1-mini) | `98c58048572b44839dfcfa60de3ad7e365a5b232` | TEA checkpoint：MIT；底層 Qwen 權重仍受 Apache-2.0 與 NOTICE／歸屬要求約束 |
| `Qwen3-ForcedAligner-0.6B` | [Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) | `c7cbfc2048c462b0d63a45797104fc9db3ad62b7` | Apache-2.0 |

本 Release 不改變任何上游模型授權。使用、修改或再散布前，請閱讀各模型封裝內的 README 及原始模型頁。

## 未收錄 pyannote

`pyannote/speaker-diarization-community-1` 要求每位下載者先在 Hugging Face 接受存取條件，因此不會鏡像、加密或附加到 GitHub Release。請在原始模型頁接受條款並使用自己的唯讀 HF Token 下載：

- [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
- [Hugging Face Access Tokens](https://huggingface.co/settings/tokens)

## 還原方式

將 Release 中的 `restore_release_models.ps1` 與 `model-release-manifest.json` 放在專案根目錄，執行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\restore_release_models.ps1 -ReleaseTag models-v2
```

若不另行指定，模型會還原到專案的 `models` 資料夾。加入 `-KeepDownloads` 可保留已驗證的 Release 分片與重組封裝。

`models-v1` 是舊版封裝，仍包含 2B `TEA-ASR-1.1`；它不符合現行 app 預設的 `models\TEA-ASR-1.1-mini` 路徑。
