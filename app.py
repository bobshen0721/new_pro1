from __future__ import annotations

import html
import json
import os
import shutil
import subprocess
import tempfile
import threading
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")

import gradio as gr
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

ROOT = Path(__file__).resolve().parent
WHISPER_DIR = ROOT / "models" / "faster-whisper-large-v3"
PYANNOTE_DIR = ROOT / "models" / "pyannote-community-1"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_CACHE: dict[tuple[str, str, str, str], tuple[WhisperModel, Pipeline]] = {}
MODEL_LOCK = threading.Lock()

PROMPT = (
    "台灣金融業電話逐字稿。內容可能包含投信、基金、ETF、淨值、申購、贖回、"
    "配息、風險報酬等級、日期、金額、百分比與證券代號。請使用繁體中文。"
)
HOTWORDS = "投信 投顧 基金 ETF 淨值 申購 贖回 配息 除息 風險報酬等級 證券代號"

PROFILES = {
    "平衡（建議先用）": (0.35, 100, 450, 250),
    "雜音較多": (0.45, 130, 500, 250),
    "遠端聲音較小": (0.28, 80, 400, 300),
}

CSS = """
.transcript-list{display:flex;flex-direction:column;gap:8px}
.transcript-row{width:100%;display:grid;grid-template-columns:90px 85px 1fr;gap:10px;
padding:12px;border:1px solid var(--border-color-primary);border-radius:10px;
background:var(--background-fill-secondary);color:var(--body-text-color);text-align:left;cursor:pointer}
.transcript-row:hover,.transcript-row.active{border-color:var(--color-accent);background:var(--background-fill-primary)}
.transcript-row.overlap{border-left:5px solid #d97706}.time,.speaker{font-weight:700}
.overlap-label{margin-right:7px;padding:1px 7px;border-radius:999px;background:#fef3c7;color:#92400e;font-size:.85em}
.empty{padding:25px;border:1px dashed var(--border-color-primary);border-radius:10px;text-align:center}
"""

JS = r"""
function installSeek(){
 const app=document.querySelector('gradio-app');
 const root=(app&&app.shadowRoot)?app.shadowRoot:document;
 if(root.__seekInstalled)return; root.__seekInstalled=true;
 root.addEventListener('click',function(e){
   const row=e.target.closest('.transcript-row'); if(!row)return;
   const audio=root.querySelector('#audio_player audio')||document.querySelector('#audio_player audio');
   if(!audio){alert('找不到播放器');return;}
   root.querySelectorAll('.transcript-row.active').forEach(x=>x.classList.remove('active'));
   row.classList.add('active'); audio.currentTime=Number(row.dataset.start||0); audio.play().catch(()=>{});
 });
}
setTimeout(installSeek,300);setTimeout(installSeek,1000);setInterval(installSeek,3000);
"""


def overlap_seconds(a1: float, a2: float, b1: float, b2: float) -> float:
    return max(0.0, min(a2, b2) - max(a1, b1))


def time_text(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes:02d}:{seconds - minutes * 60:05.2f}"


def get_device(device_ui: str, compute_ui: str) -> tuple[str, str]:
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device_ui == "自動" else device_ui
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PyTorch 找不到 CUDA，請改用 CPU。")
    compute = ("float16" if device == "cuda" else "int8") if compute_ui == "自動" else compute_ui
    if device == "cpu" and compute == "float16":
        compute = "int8"
    return device, compute


def load_models(whisper_path: str, pyannote_path: str, device_ui: str, compute_ui: str):
    whisper_dir = Path(whisper_path).expanduser().resolve()
    pyannote_dir = Path(pyannote_path).expanduser().resolve()
    if not whisper_dir.is_dir():
        raise FileNotFoundError(f"找不到 faster-whisper 模型：{whisper_dir}")
    if not pyannote_dir.is_dir():
        raise FileNotFoundError(f"找不到 pyannote 模型：{pyannote_dir}")

    device, compute = get_device(device_ui, compute_ui)
    key = (str(whisper_dir), str(pyannote_dir), device, compute)
    with MODEL_LOCK:
        if key not in MODEL_CACHE:
            whisper = WhisperModel(str(whisper_dir), device=device, compute_type=compute, local_files_only=True)
            diarizer = Pipeline.from_pretrained(str(pyannote_dir))
            if device == "cuda":
                diarizer.to(torch.device("cuda"))
            MODEL_CACHE[key] = (whisper, diarizer)
    return *MODEL_CACHE[key], device, compute


def normalize_audio(source: str, workdir: Path) -> Path:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("找不到 FFmpeg，請先安裝並加入 PATH。")
    output = workdir / "normalized.wav"
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", source,
               "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return output


def whisper_words(model: WhisperModel, audio: Path, threshold: float, speech_ms: int,
                  silence_ms: int, pad_ms: int, prompt: str, hotwords: str):
    segments, info = model.transcribe(
        str(audio), language="zh", beam_size=5, temperature=(0.0, 0.2, 0.4),
        condition_on_previous_text=False, word_timestamps=True,
        compression_ratio_threshold=2.4, log_prob_threshold=-1.0, no_speech_threshold=0.6,
        hallucination_silence_threshold=1.0, initial_prompt=prompt.strip() or None,
        hotwords=hotwords.strip() or None, vad_filter=True,
        vad_parameters={
            "threshold": float(threshold), "neg_threshold": max(0.01, float(threshold) - 0.15),
            "min_speech_duration_ms": int(speech_ms), "min_silence_duration_ms": int(silence_ms),
            "speech_pad_ms": int(pad_ms), "max_speech_duration_s": 28,
        },
    )
    words = []
    for segment in segments:
        for word in segment.words or []:
            if word.start is not None and word.end is not None:
                words.append({"start": float(word.start), "end": float(word.end),
                              "text": word.word, "probability": float(word.probability)})
    info_dict = {"language": info.language, "language_probability": float(info.language_probability),
                 "duration": float(info.duration), "duration_after_vad": float(info.duration_after_vad),
                 "word_count": len(words)}
    return words, info_dict


def annotation_turns(annotation: Any) -> list[dict[str, Any]]:
    turns = []
    for segment, _, label in annotation.itertracks(yield_label=True):
        turns.append({"start": float(segment.start), "end": float(segment.end), "label": str(label)})
    return sorted(turns, key=lambda x: (x["start"], x["end"]))


def aliases_from(turns: list[dict[str, Any]]) -> dict[str, str]:
    labels = []
    for turn in turns:
        if turn["label"] not in labels:
            labels.append(turn["label"])
    return {label: chr(ord("A") + i) for i, label in enumerate(labels)}


def overlap_regions(annotation: Any, minimum: float) -> list[dict[str, float]]:
    regions = []
    for segment in annotation.get_overlap().support():
        duration = float(segment.end - segment.start)
        if duration >= minimum:
            regions.append({"start": float(segment.start), "end": float(segment.end), "duration": duration})
    return regions


def choose_speaker(start: float, end: float, turns: list[dict[str, Any]]) -> str:
    scores: dict[str, float] = defaultdict(float)
    for turn in turns:
        scores[turn["label"]] += overlap_seconds(start, end, turn["start"], turn["end"])
    if scores and max(scores.values()) > 0:
        return max(scores, key=scores.get)
    middle = (start + end) / 2
    nearest = min(turns, key=lambda x: min(abs(middle - x["start"]), abs(middle - x["end"])), default=None)
    return nearest["label"] if nearest else "UNKNOWN"


def merge_results(words, turns, overlaps, aliases, minimum_ratio):
    tagged = []
    for word in words:
        label = choose_speaker(word["start"], word["end"], turns)
        duration = max(0.01, word["end"] - word["start"])
        overlap = sum(overlap_seconds(word["start"], word["end"], x["start"], x["end"]) for x in overlaps)
        tagged.append({**word, "speaker": aliases.get(label, "?"),
                       "suspected_overlap": overlap >= 0.05 and overlap / duration >= minimum_ratio})

    rows = []
    for word in tagged:
        can_merge = rows and rows[-1]["speaker"] == word["speaker"] and \
                    rows[-1]["suspected_overlap"] == word["suspected_overlap"] and \
                    word["start"] - rows[-1]["end"] <= 0.8
        if can_merge:
            rows[-1]["end"] = word["end"]
            rows[-1]["text"] += word["text"]
        else:
            rows.append({"start": word["start"], "end": word["end"], "speaker": word["speaker"],
                         "suspected_overlap": word["suspected_overlap"], "text": word["text"]})
    for row in rows:
        row["text"] = row["text"].strip()
    return [row for row in rows if row["text"]]


def to_html(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<div class="empty">沒有辨識到逐字稿。</div>'
    result = ['<div class="transcript-list">']
    for row in rows:
        cls = " overlap" if row["suspected_overlap"] else ""
        badge = '<span class="overlap-label">疑似同時說話</span>' if row["suspected_overlap"] else ""
        result.append(
            f'<button type="button" class="transcript-row{cls}" data-start="{row["start"]:.3f}">'
            f'<span class="time">{time_text(row["start"])}</span>'
            f'<span class="speaker">說話者 {html.escape(row["speaker"])}</span>'
            f'<span>{badge}{html.escape(row["text"])}</span></button>'
        )
    return "".join(result) + "</div>"


def save_result(result: dict[str, Any], source_name: str) -> list[str]:
    folder = OUTPUT_DIR / uuid.uuid4().hex
    folder.mkdir(parents=True)
    stem = Path(source_name).stem or "transcript"
    json_path = folder / f"{stem}.json"
    txt_path = folder / f"{stem}.txt"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = []
    for row in result["transcript"]:
        flag = "【疑似同時說話】" if row["suspected_overlap"] else ""
        lines.append(f"[{time_text(row['start'])} - {time_text(row['end'])}] {flag}說話者 {row['speaker']}：{row['text']}")
    txt_path.write_text("\n".join(lines), encoding="utf-8-sig")
    return [str(json_path), str(txt_path)]


def process(audio_path, whisper_path, pyannote_path, device_ui, compute_ui, threshold,
            speech_ms, silence_ms, pad_ms, min_overlap, overlap_ratio, prompt, hotwords,
            progress=gr.Progress()):
    if not audio_path:
        return "請先上傳音檔。", to_html([]), None, None
    try:
        progress(0.05, desc="載入離線模型")
        whisper, diarizer, device, compute = load_models(whisper_path, pyannote_path, device_ui, compute_ui)
        with tempfile.TemporaryDirectory() as tmp:
            progress(0.12, desc="統一音檔格式")
            normalized = normalize_audio(audio_path, Path(tmp))
            progress(0.25, desc="產生逐字稿")
            words, info = whisper_words(whisper, normalized, threshold, speech_ms, silence_ms, pad_ms, prompt, hotwords)
            progress(0.65, desc="判斷說話者 A/B")
            output = diarizer(str(normalized), num_speakers=2)
            regular = output.speaker_diarization
            exclusive = getattr(output, "exclusive_speaker_diarization", None) or regular
            turns = annotation_turns(exclusive)
            aliases = aliases_from(turns)
            progress(0.82, desc="標記重疊語音")
            overlaps = overlap_regions(regular, float(min_overlap))
            transcript = merge_results(words, turns, overlaps, aliases, float(overlap_ratio))
            result = {"audio_file": Path(audio_path).name, "device": device, "compute_type": compute,
                      "whisper_info": info, "speaker_mapping": aliases, "overlap_regions": overlaps,
                      "speaker_turns": turns, "transcript": transcript}
            files = save_result(result, Path(audio_path).name)
        count = sum(x["suspected_overlap"] for x in transcript)
        progress(1.0, desc="完成")
        return f"完成：{len(transcript)} 段，其中 {count} 段疑似同時說話。", to_html(transcript), files, result
    except Exception as exc:
        return f"處理失敗：{type(exc).__name__}: {exc}", to_html([]), None, None


def apply_profile(name):
    return PROFILES[name]


with gr.Blocks(css=CSS, js=JS, title="台灣金融業電話逐字稿") as demo:
    gr.Markdown("# 台灣金融業電話逐字稿\n"
                "faster-whisper + Silero VAD + pyannote + 重疊語音標記。點擊逐字稿可跳到該時間播放。")
    with gr.Row():
        audio = gr.Audio(label="上傳單聲道電話錄音", sources=["upload"], type="filepath", elem_id="audio_player")
        with gr.Column():
            profile = gr.Dropdown(list(PROFILES), value="平衡（建議先用）", label="錄音情境")
            device = gr.Dropdown(["自動", "cuda", "cpu"], value="自動", label="運算裝置")
            compute = gr.Dropdown(["自動", "float16", "int8_float16", "int8"], value="自動", label="計算格式")
            start = gr.Button("開始產生逐字稿", variant="primary")
    with gr.Accordion("進階參數", open=False):
        with gr.Row():
            threshold = gr.Slider(0.1, 0.8, 0.35, step=0.01, label="VAD 語音門檻")
            speech_ms = gr.Slider(40, 500, 100, step=10, label="最短語音（毫秒）")
            silence_ms = gr.Slider(100, 2000, 450, step=50, label="切段靜音（毫秒）")
            pad_ms = gr.Slider(0, 600, 250, step=25, label="前後保留（毫秒）")
        with gr.Row():
            min_overlap = gr.Slider(0.05, 1.0, 0.20, step=0.05, label="最短重疊區間（秒）")
            overlap_ratio = gr.Slider(0.05, 0.9, 0.25, step=0.05, label="字詞重疊比例")
        prompt = gr.Textbox(PROMPT, label="金融提示文字", lines=3)
        hotwords = gr.Textbox(HOTWORDS, label="金融關鍵詞", lines=2)
    with gr.Accordion("離線模型路徑", open=False):
        whisper_path = gr.Textbox(str(WHISPER_DIR), label="faster-whisper 模型資料夾")
        pyannote_path = gr.Textbox(str(PYANNOTE_DIR), label="pyannote 模型資料夾")
    status = gr.Markdown("尚未開始。")
    gr.Markdown("## 可點擊逐字稿")
    transcript_html = gr.HTML('<div class="empty">完成後顯示在這裡。</div>')
    downloads = gr.File(label="下載 JSON 與 TXT", file_count="multiple")
    with gr.Accordion("原始 JSON", open=False):
        raw_json = gr.JSON()

    profile.change(apply_profile, profile, [threshold, speech_ms, silence_ms, pad_ms], api_name=False)
    start.click(process,
                [audio, whisper_path, pyannote_path, device, compute, threshold, speech_ms,
                 silence_ms, pad_ms, min_overlap, overlap_ratio, prompt, hotwords],
                [status, transcript_html, downloads, raw_json], concurrency_limit=1)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=7860, show_error=True)
