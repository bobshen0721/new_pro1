# TEA-ASR mini Release 說明

GitHub Release `models-v2` 只提供現行應用程式新增的 `TEA-ASR-1.1-mini`。模型以未壓縮 `tar` 封裝成單一小於 2 GiB 的分片；`restore_release_models.ps1` 會驗證分片 SHA-256、重組封裝、再次驗證完整封裝，最後解壓到本機 `models` 資料夾。

## 收錄模型

| 本機資料夾 | 原始來源 | 固定 revision | 授權 |
|---|---|---|---|
| `TEA-ASR-1.1-mini` | [JacobLinCool/TEA-ASR-1.1-mini](https://huggingface.co/JacobLinCool/TEA-ASR-1.1-mini) | `98c58048572b44839dfcfa60de3ad7e365a5b232` | TEA checkpoint：MIT；底層 Qwen 權重仍受 Apache-2.0 與 NOTICE／歸屬要求約束 |

本 Release 不改變上游模型授權。使用、修改或再散布前，請閱讀模型封裝內的 README 及原始模型頁。

## 未重複收錄其他模型

- Whisper large-v2 與 Qwen3-ForcedAligner-0.6B 已存在 GitHub Release `models-v1`，因此不在 `models-v2` 重複上傳。
- `pyannote/speaker-diarization-community-1` 要求每位下載者先在 Hugging Face 接受存取條件，因此不會鏡像到 GitHub Release。
- 全新環境最簡單的做法是使用 `download_models.ps1`，從原始 Hugging Face 來源下載四個必要模型。

## 還原方式

將 Release 中的 `restore_release_models.ps1` 與 `model-release-manifest.json` 放在專案根目錄，執行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\restore_release_models.ps1 -ReleaseTag models-v2
```

若不另行指定，mini 會還原到 `models\TEA-ASR-1.1-mini`。加入 `-KeepDownloads` 可保留已驗證的 Release 分片與重組封裝。

`models-v1` 仍包含舊的 2B `TEA-ASR-1.1`；現行 app 不使用該資料夾，也不能直接將它改名成 mini。
