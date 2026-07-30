# 台灣金融業電話逐字稿

這是一個適合單聲道、雙人、音質較差電話錄音的本機 Gradio 系統。

## 功能

- faster-whisper `large-v3`：中文逐字稿與逐字時間。
- faster-whisper 內建 Silero VAD：過濾靜音並改善電話切段。
- pyannote Community-1：分出說話者 A／B。
- 重疊語音標記：將疑似兩人同時說話的段落標示出來。
- 點擊逐字稿：播放器直接跳到該段開始時間並播放。
- 下載 JSON 與 TXT。
- 模型全程讀取本機資料夾，可在公司內網離線執行。

> 單聲道的兩個聲音已混在一起。系統可以標記「疑似同時說話」，但無法保證把重疊時兩個人的內容完整拆開。

## 專案檔案

```text
new_pro1/
├─ app.py
├─ requirements.txt
├─ run.bat
├─ download_models.ps1
├─ TUNING_GUIDE.md
├─ models/                 # 不上傳 GitHub
│  ├─ faster-whisper-large-v3/
│  └─ pyannote-community-1/
└─ outputs/                # 不上傳 GitHub
```

## 一、在可連外的電腦下載模型

先到 Hugging Face 接受 `pyannote/speaker-diarization-community-1` 的使用條款，再建立一個新的權杖。不要把權杖寫進程式或上傳 GitHub。

在 PowerShell 執行：

```powershell
$env:HF_TOKEN="hf_你的新權杖"
.\download_models.ps1
```

完成後，把整個 `models` 資料夾帶進公司環境。預設路徑如下：

```text
models\faster-whisper-large-v3
models\pyannote-community-1
```

## 二、公司電腦安裝

建議使用 Python 3.11，並先安裝 FFmpeg，確認下列指令有結果：

```powershell
ffmpeg -version
```

建立環境：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### NVIDIA 顯示卡

先依 PyTorch 官方安裝頁選擇 Windows、Pip 與適合的 CUDA 版本，再安裝本專案套件。例如：

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### 只用 CPU

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 三、啟動

雙擊：

```text
run.bat
```

或在 PowerShell 執行：

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
4. 點擊任一段逐字稿，音檔會跳到該段開始時間播放。
5. 下載 JSON 或 TXT。

## A／B 的意思

A 是該通電話中最早被模型辨識到的說話者，B 是另一位。A 不一定永遠是客服；下一通電話會重新判斷。

## 安全注意事項

- 不要把電話錄音、模型、輸出結果或權杖提交到 GitHub。
- 本專案預設只監聽 `127.0.0.1`，不會直接開放給其他電腦。
- 金額、日期、證券代號與疑似重疊段落，仍應交由人工複核。
- 金融業正式使用前，應依公司個資、錄音保存、存取權限及稽核規範評估。

## 調參

請看 [TUNING_GUIDE.md](TUNING_GUIDE.md)。
