# 台灣金融業電話逐字稿

適合單聲道、雙人及音質較差電話錄音的本機 Gradio 系統。預設完全離線執行，錄音不會送到外部服務。

## 模型與處理流程

1. Silero 語音偵測：找出有人說話的區間，保留原始時間位置。
2. `TEA-ASR-1.1`：產生台灣繁體中文主逐字稿。
3. `Qwen3-ForcedAligner-0.6B`：替主逐字稿產生字／詞時間戳。
4. `faster-whisper large-v2`：產生第二份比對稿，不會自動改寫主稿。
5. `pyannote Community-1`：分出說話者 A／B，並標示疑似重疊語音。
6. 文字相似度：先統一繁簡體、大小寫及標點，再計算兩份完整文字的編輯距離相似度。

兩個語音辨識模型不會被硬拼成一句話。相似度達 90% 時採用 TEA 主稿；低於 90% 時仍顯示 TEA 主稿，但明確提醒人工再檢查。

## 功能

- 台灣繁體中文主逐字稿及 Whisper large-v2 比對稿。
- 顯示兩個模型的文字相似度，低於 90% 自動提出人工複核警告。
- Qwen3-ForcedAligner 字／詞時間戳。
- Pyannote Community-1 說話者 A／B 分離。
- 疑似兩人同時說話標記。
- 點擊逐字稿可讓播放器跳到該段開始時間。
- 下載 JSON 與 TXT；JSON 保留兩個模型原文、語音區間及說話者資料。
- 模型全程讀取本機資料夾，適合無法連 Hugging Face 的公司環境。

> 單聲道中兩個人的聲音已混在一起。系統可以判斷誰在何時說話及標記重疊，但無法保證把重疊時兩人的內容完整拆開。

## 專案檔案

```text
new_pro1/
├─ app.py
├─ requirements.txt
├─ run.bat
├─ download_models.ps1
├─ TUNING_GUIDE.md
├─ models/                             # 不上傳 GitHub
│  ├─ faster-whisper-large-v2/
│  ├─ TEA-ASR-1.1/
│  ├─ Qwen3-ForcedAligner-0.6B/
│  └─ pyannote-community-1/
└─ outputs/                            # 不上傳 GitHub
```

## 一、在可連外的電腦下載模型

先在 Hugging Face 接受 [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) 的使用條款，再建立一個唯讀權杖。不要把權杖寫進程式或上傳 GitHub。

在 PowerShell 執行：

```powershell
$env:HF_TOKEN="hf_你的唯讀權杖"
.\download_models.ps1
```

腳本會下載：

- `Systran/faster-whisper-large-v2`
- `JacobLinCool/TEA-ASR-1.1`
- `Qwen/Qwen3-ForcedAligner-0.6B`
- `pyannote/speaker-diarization-community-1`

完成後，把整個 `models` 資料夾帶進公司環境。

## 二、Windows 11 安裝

建議使用 Python 3.11，並先安裝 FFmpeg：

```powershell
ffmpeg -version
```

建立虛擬環境：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### NVIDIA 顯示卡

先依 [PyTorch 官方安裝頁](https://pytorch.org/get-started/locally/) 選擇 Windows、Pip 與適合的 CUDA 版本，再安裝本專案套件。例如：

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

RTX 3090 24GB 可使用預設設定。程式會依序推論，不會同時執行四個模型；模型仍會常駐顯示卡記憶體，以減少下一通錄音的等待時間。

### 只使用中央處理器

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

TEA-ASR、ForcedAligner 與 Pyannote 在中央處理器上會非常慢，只建議用來排除安裝問題。

## 三、啟動

雙擊 `run.bat`，或在 PowerShell 執行：

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

瀏覽器開啟：

```text
http://127.0.0.1:7860
```

## 使用方式

1. 上傳電話錄音。
2. 先選「平衡（建議先用）」。
3. 按下「開始產生逐字稿」。
4. 以 TEA-ASR 主逐字稿為底稿，對照下方 Whisper large-v2 結果。
5. 若相似度低於 90%，回放音訊並人工檢查兩份結果的差異。
6. 點擊任一主逐字稿段落，可跳到音檔原始時間播放。
7. 下載 JSON 或 TXT。

## A／B 的意思

A 是該通電話中最早被模型辨識到的說話者，B 是另一位。A 不一定永遠是客服，每通電話都會重新判斷。

## 離線與安全注意事項

- 啟動時會設定 Hugging Face 與 Transformers 離線模式；四個模型必須已完整下載。
- 不要把電話錄音、模型、輸出結果或權杖提交到 GitHub。
- 服務預設只監聽 `127.0.0.1`，不會直接開放給其他電腦。
- 金額、日期、帳號、證券代號及疑似重疊段落，正式使用前必須人工複核。
- 導入金融業正式環境前，應依公司個資、錄音保存、權限及稽核規範評估。

## 調整參數

請看 [TUNING_GUIDE.md](TUNING_GUIDE.md)。
