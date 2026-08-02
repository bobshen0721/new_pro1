from __future__ import annotations

import gc
import html
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")

import gradio as gr
import torch
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps
from pyannote.audio import Pipeline
from qwen_asr import Qwen3ASRModel

from comparison import SIMILARITY_THRESHOLD, append_text, join_texts, score_time_aligned_segments

ROOT = Path(__file__).resolve().parent
WHISPER_DIR = ROOT / "models" / "faster-whisper-large-v2"
TEA_DIR = ROOT / "models" / "TEA-ASR-1.1"
ALIGNER_DIR = ROOT / "models" / "Qwen3-ForcedAligner-0.6B"
PYANNOTE_DIR = ROOT / "models" / "pyannote-community-1"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class ModelBundle:
    whisper: WhisperModel
    tea: Qwen3ASRModel
    diarizer: Pipeline
    device: str
    whisper_compute: str
    qwen_dtype: str


MODEL_CACHE_KEY: tuple[str, ...] | None = None
MODEL_CACHE: ModelBundle | None = None
MODEL_LOCK = threading.Lock()

PROMPT = (
    "台灣金融業客服電話逐字稿。內容可能包含投信、投顧、基金、ETF、淨值、申購、贖回、"
    "配息、風險報酬等級、日期、金額、百分比與證券代號。請忠實轉錄，不要摘要或補寫。"
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
.transcript-row.overlap{border-left:5px solid #d97706}
.transcript-row.review .transcript-text{color:#dc2626;font-weight:700}
.time,.speaker{font-weight:700}.transcript-content,.transcript-text{display:block}
.badges{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:5px}
.overlap-label{margin-right:7px;padding:1px 7px;border-radius:999px;background:#fef3c7;color:#92400e;font-size:.85em}
.score-label{font-size:.85em;font-weight:700;color:var(--body-text-color-subdued)}
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
    if device == "cpu" and compute in {"float16", "int8_float16"}:
        compute = "int8"
    return device, compute


def _required_model_dir(path: str, model_name: str) -> Path:
    model_dir = Path(path).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"找不到 {model_name} 模型：{model_dir}")
    return model_dir


def load_models(
    whisper_path: str,
    tea_path: str,
    aligner_path: str,
    pyannote_path: str,
    device_ui: str,
    compute_ui: str,
) -> ModelBundle:
    global MODEL_CACHE_KEY, MODEL_CACHE

    whisper_dir = _required_model_dir(whisper_path, "Whisper large-v2")
    tea_dir = _required_model_dir(tea_path, "TEA-ASR-1.1")
    aligner_dir = _required_model_dir(aligner_path, "Qwen3-ForcedAligner")
    pyannote_dir = _required_model_dir(pyannote_path, "Pyannote Community-1")
    device, compute = get_device(device_ui, compute_ui)
    key = (
        str(whisper_dir),
        str(tea_dir),
        str(aligner_dir),
        str(pyannote_dir),
        device,
        compute,
    )

    with MODEL_LOCK:
        if MODEL_CACHE is not None and MODEL_CACHE_KEY == key:
            return MODEL_CACHE

        MODEL_CACHE = None
        MODEL_CACHE_KEY = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        whisper = WhisperModel(
            str(whisper_dir),
            device=device,
            compute_type=compute,
            local_files_only=True,
        )

        if device == "cuda":
            supports_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            qwen_dtype = torch.bfloat16 if supports_bf16 else torch.float16
            qwen_device = "cuda:0"
        else:
            qwen_dtype = torch.float32
            qwen_device = "cpu"

        tea = Qwen3ASRModel.from_pretrained(
            str(tea_dir),
            dtype=qwen_dtype,
            device_map=qwen_device,
            forced_aligner=str(aligner_dir),
            forced_aligner_kwargs={"dtype": qwen_dtype, "device_map": qwen_device},
            max_inference_batch_size=4 if device == "cuda" else 1,
            max_new_tokens=512,
        )

        diarizer = Pipeline.from_pretrained(str(pyannote_dir))
        if device == "cuda":
            diarizer.to(torch.device("cuda"))

        MODEL_CACHE = ModelBundle(
            whisper=whisper,
            tea=tea,
            diarizer=diarizer,
            device=device,
            whisper_compute=compute,
            qwen_dtype=str(qwen_dtype).replace("torch.", ""),
        )
        MODEL_CACHE_KEY = key
        return MODEL_CACHE


def normalize_audio(source: str, workdir: Path) -> Path:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("找不到 FFmpeg，請先安裝並加入 PATH。")
    output = workdir / "normalized.wav"
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        source,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return output


def detect_speech(
    audio: Path,
    threshold: float,
    speech_ms: int,
    silence_ms: int,
    pad_ms: int,
) -> tuple[Any, list[dict[str, Any]]]:
    samples = decode_audio(str(audio), sampling_rate=16000)
    options = VadOptions(
        threshold=float(threshold),
        neg_threshold=max(0.01, float(threshold) - 0.15),
        min_speech_duration_ms=int(speech_ms),
        min_silence_duration_ms=int(silence_ms),
        speech_pad_ms=int(pad_ms),
        max_speech_duration_s=28,
    )
    timestamps = get_speech_timestamps(samples, vad_options=options, sampling_rate=16000)
    regions = [
        {
            "start_sample": int(item["start"]),
            "end_sample": int(item["end"]),
            "start": round(float(item["start"]) / 16000.0, 3),
            "end": round(float(item["end"]) / 16000.0, 3),
        }
        for item in timestamps
        if int(item["end"]) > int(item["start"])
    ]
    return samples, regions


def tea_transcript(
    model: Qwen3ASRModel,
    samples: Any,
    regions: list[dict[str, Any]],
    context: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not regions:
        return [], {"language": "Chinese", "text": "", "chunk_count": 0, "aligned_item_count": 0}

    audio_chunks = [
        (samples[region["start_sample"] : region["end_sample"]], 16000)
        for region in regions
    ]
    results = model.transcribe(
        audio=audio_chunks,
        context=context,
        language="Chinese",
        return_time_stamps=True,
    )
    if len(results) != len(regions):
        raise RuntimeError("TEA-ASR 回傳的分段數量不一致。")

    aligned_items: list[dict[str, Any]] = []
    chunk_results: list[dict[str, Any]] = []
    for region, result in zip(regions, results):
        text = (result.text or "").strip()
        timestamps = result.time_stamps or []
        if text and not timestamps:
            raise RuntimeError("TEA-ASR 已產生文字，但 Qwen3-ForcedAligner 沒有回傳時間戳。")
        offset = float(region["start"])
        for item in timestamps:
            item_text = str(item.text)
            if not item_text:
                continue
            start = max(0.0, offset + float(item.start_time))
            end = max(start, offset + float(item.end_time))
            aligned_items.append(
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "text": item_text,
                    "timestamp_source": "Qwen3-ForcedAligner-0.6B",
                }
            )
        chunk_results.append(
            {
                "start": region["start"],
                "end": region["end"],
                "language": result.language,
                "text": text,
                "aligned_item_count": len(timestamps),
            }
        )

    aligned_items.sort(key=lambda item: (item["start"], item["end"]))
    info = {
        "language": "Chinese",
        "text": join_texts([item["text"] for item in chunk_results]),
        "chunk_count": len(chunk_results),
        "aligned_item_count": len(aligned_items),
        "chunks": chunk_results,
    }
    return aligned_items, info


def whisper_transcript(
    model: WhisperModel,
    audio: Path,
    threshold: float,
    speech_ms: int,
    silence_ms: int,
    pad_ms: int,
    prompt: str,
    hotwords: str,
) -> dict[str, Any]:
    segments, info = model.transcribe(
        str(audio),
        language="zh",
        beam_size=5,
        temperature=(0.0, 0.2, 0.4),
        condition_on_previous_text=False,
        word_timestamps=True,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        hallucination_silence_threshold=1.0,
        initial_prompt=prompt.strip() or None,
        hotwords=hotwords.strip() or None,
        vad_filter=True,
        vad_parameters={
            "threshold": float(threshold),
            "neg_threshold": max(0.01, float(threshold) - 0.15),
            "min_speech_duration_ms": int(speech_ms),
            "min_silence_duration_ms": int(silence_ms),
            "speech_pad_ms": int(pad_ms),
            "max_speech_duration_s": 28,
        },
    )
    segment_rows = []
    word_rows = []
    for segment in segments:
        segment_text = segment.text.strip()
        if segment_text:
            segment_rows.append(
                {"start": float(segment.start), "end": float(segment.end), "text": segment_text}
            )
        for word in segment.words or []:
            word_text = str(word.word).strip()
            if not word_text or word.start is None or word.end is None:
                continue
            word_rows.append(
                {"start": float(word.start), "end": float(word.end), "text": word_text}
            )
    return {
        "language": info.language,
        "language_probability": float(info.language_probability),
        "duration": float(info.duration),
        "duration_after_vad": float(getattr(info, "duration_after_vad", info.duration)),
        "text": join_texts([item["text"] for item in segment_rows]),
        "segments": segment_rows,
        "words": word_rows,
    }


def annotation_turns(annotation: Any) -> list[dict[str, Any]]:
    turns = []
    for segment, _, label in annotation.itertracks(yield_label=True):
        turns.append({"start": float(segment.start), "end": float(segment.end), "label": str(label)})
    return sorted(turns, key=lambda item: (item["start"], item["end"]))


def aliases_from(turns: list[dict[str, Any]]) -> dict[str, str]:
    labels = []
    for turn in turns:
        if turn["label"] not in labels:
            labels.append(turn["label"])
    return {label: chr(ord("A") + index) for index, label in enumerate(labels)}


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
    nearest = min(
        turns,
        key=lambda item: min(abs(middle - item["start"]), abs(middle - item["end"])),
        default=None,
    )
    return nearest["label"] if nearest else "UNKNOWN"


def merge_results(words, turns, overlaps, aliases, minimum_ratio):
    tagged = []
    for word in words:
        label = choose_speaker(word["start"], word["end"], turns)
        duration = max(0.01, word["end"] - word["start"])
        overlap = sum(
            overlap_seconds(word["start"], word["end"], region["start"], region["end"])
            for region in overlaps
        )
        tagged.append(
            {
                **word,
                "speaker": aliases.get(label, "?"),
                "suspected_overlap": overlap >= 0.05 and overlap / duration >= minimum_ratio,
            }
        )

    rows = []
    for word in tagged:
        can_merge = (
            rows
            and rows[-1]["speaker"] == word["speaker"]
            and rows[-1]["suspected_overlap"] == word["suspected_overlap"]
            and word["start"] - rows[-1]["end"] <= 0.8
        )
        if can_merge:
            rows[-1]["end"] = word["end"]
            rows[-1]["text"] = append_text(rows[-1]["text"], word["text"])
        else:
            rows.append(
                {
                    "start": word["start"],
                    "end": word["end"],
                    "speaker": word["speaker"],
                    "suspected_overlap": word["suspected_overlap"],
                    "text": word["text"],
                }
            )
    for row in rows:
        row["text"] = row["text"].strip()
    return [row for row in rows if row["text"]]


def to_html(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<div class="empty">沒有辨識到逐字稿。</div>'
    result = ['<div class="transcript-list">']
    for row in rows:
        css_class = " overlap" if row["suspected_overlap"] else ""
        needs_review = bool(row.get("needs_manual_review", False))
        if needs_review:
            css_class += " review"
        badges = []
        if row["suspected_overlap"]:
            badges.append('<span class="overlap-label">疑似同時說話</span>')
        similarity_percent = float(row.get("similarity_percent", 0.0))
        badges.append(f'<span class="score-label">相似度 {similarity_percent:.2f}%</span>')
        result.append(
            f'<button type="button" class="transcript-row{css_class}" data-start="{row["start"]:.3f}">'
            f'<span class="time">{time_text(row["start"])}</span>'
            f'<span class="speaker">說話者 {html.escape(row["speaker"])}</span>'
            f'<span class="transcript-content"><span class="badges">{"".join(badges)}</span>'
            f'<span class="transcript-text">{html.escape(row["text"])}</span></span></button>'
        )
    return "".join(result) + "</div>"


def safe_stem(source_name: str) -> str:
    stem = Path(source_name).stem or "transcript"
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", stem).strip("._")
    return cleaned or "transcript"


def save_result(result: dict[str, Any], source_name: str) -> list[str]:
    folder = OUTPUT_DIR / uuid.uuid4().hex
    folder.mkdir(parents=True)
    stem = safe_stem(source_name)
    json_path = folder / f"{stem}.json"
    txt_path = folder / f"{stem}.txt"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    comparison = result["model_comparison"]
    review_count = int(comparison["manual_review_segment_count"])
    review_text = f"{review_count} 段需人工檢查" if review_count else "所有段落均達 90%"
    lines = [
        "TEA-ASR-1.1 主逐字稿（Qwen3-ForcedAligner 時間戳）",
        f"時間對齊後的加權相似度：{comparison['similarity_percent']:.2f}%（{review_text}）",
        "",
    ]
    for row in result["transcript"]:
        flags = []
        if row["suspected_overlap"]:
            flags.append("【疑似同時說話】")
        flags.append(f"【相似度 {row['similarity_percent']:.2f}%】")
        lines.append(
            f"[{time_text(row['start'])} - {time_text(row['end'])}] "
            f"{''.join(flags)}說話者 {row['speaker']}：{row['text']}"
        )
    lines.extend(["", "Whisper large-v2 比對稿（不自動覆蓋主稿）", "", result["whisper_v2"]["text"]])
    txt_path.write_text("\n".join(lines), encoding="utf-8-sig")
    return [str(json_path), str(txt_path)]


def process(
    audio_path,
    whisper_path,
    tea_path,
    aligner_path,
    pyannote_path,
    device_ui,
    compute_ui,
    threshold,
    speech_ms,
    silence_ms,
    pad_ms,
    min_overlap,
    overlap_ratio,
    prompt,
    hotwords,
    progress=gr.Progress(),
):
    if not audio_path:
        return "請先上傳音檔。", to_html([]), "", None, None
    try:
        progress(0.03, desc="載入四個離線模型")
        bundle = load_models(
            whisper_path,
            tea_path,
            aligner_path,
            pyannote_path,
            device_ui,
            compute_ui,
        )
        with tempfile.TemporaryDirectory() as tmp:
            progress(0.10, desc="統一音檔格式")
            normalized = normalize_audio(audio_path, Path(tmp))

            progress(0.16, desc="偵測語音區間")
            samples, speech_regions = detect_speech(
                normalized,
                threshold,
                speech_ms,
                silence_ms,
                pad_ms,
            )

            context = "\n".join(part for part in [prompt.strip(), hotwords.strip()] if part)
            progress(0.24, desc="TEA-ASR-1.1 產生主逐字稿與時間戳")
            aligned_items, tea_info = tea_transcript(bundle.tea, samples, speech_regions, context)

            progress(0.58, desc="Whisper large-v2 產生比對稿")
            whisper_info = whisper_transcript(
                bundle.whisper,
                normalized,
                threshold,
                speech_ms,
                silence_ms,
                pad_ms,
                prompt,
                hotwords,
            )

            progress(0.72, desc="Pyannote Community-1 判斷說話者 A/B")
            diarization_output = bundle.diarizer(str(normalized), num_speakers=2)
            regular = getattr(diarization_output, "speaker_diarization", diarization_output)
            exclusive = getattr(diarization_output, "exclusive_speaker_diarization", None)
            if exclusive is None:
                exclusive = regular
            turns = annotation_turns(exclusive)
            aliases = aliases_from(turns)

            progress(0.87, desc="依時間對齊逐段計算相似度")
            overlaps = overlap_regions(regular, float(min_overlap))
            transcript = merge_results(aligned_items, turns, overlaps, aliases, float(overlap_ratio))
            transcript, similarity, alignment_source = score_time_aligned_segments(
                transcript,
                whisper_info["words"],
                whisper_info["segments"],
            )
            review_count = sum(row["needs_manual_review"] for row in transcript)
            needs_manual_review = not transcript or review_count > 0
            result = {
                "audio_file": Path(audio_path).name,
                "pipeline": {
                    "primary_asr": "JacobLinCool/TEA-ASR-1.1",
                    "comparison_asr": "Systran/faster-whisper-large-v2",
                    "timestamps": "Qwen/Qwen3-ForcedAligner-0.6B",
                    "speakers": "pyannote/speaker-diarization-community-1",
                },
                "runtime": {
                    "device": bundle.device,
                    "whisper_compute_type": bundle.whisper_compute,
                    "qwen_dtype": bundle.qwen_dtype,
                },
                "vad_regions": speech_regions,
                "tea_asr": tea_info,
                "whisper_v2": whisper_info,
                "model_comparison": {
                    "method": "time_aligned_segment_normalized_levenshtein_similarity",
                    "alignment_source": alignment_source,
                    "similarity_percent": round(similarity * 100.0, 2),
                    "threshold_percent": SIMILARITY_THRESHOLD * 100.0,
                    "segment_count": len(transcript),
                    "manual_review_segment_count": review_count,
                    "needs_manual_review": needs_manual_review,
                },
                "speaker_mapping": aliases,
                "overlap_regions": overlaps,
                "speaker_turns": turns,
                "transcript": transcript,
            }
            files = save_result(result, Path(audio_path).name)

        overlap_count = sum(row["suspected_overlap"] for row in transcript)
        progress(1.0, desc="完成")
        similarity_percent = similarity * 100.0
        if not transcript:
            review_message = "⚠️ 沒有可評分段落，請人工檢查。"
        elif review_count:
            review_message = f"⚠️ {review_count} 段低於 90%，已用紅色標示。"
        else:
            review_message = "✅ 所有段落均達 90%。"
        status = (
            f"完成：TEA 主稿共 {len(transcript)} 段，其中 {overlap_count} 段疑似同時說話。  \n"
            f"時間對齊後的加權相似度：**{similarity_percent:.2f}%**。{review_message}"
        )
        return status, to_html(transcript), whisper_info["text"], files, result
    except Exception as exc:
        return f"處理失敗：{type(exc).__name__}: {exc}", to_html([]), "", None, None


def apply_profile(name):
    return PROFILES[name]


with gr.Blocks(css=CSS, js=JS, title="台灣金融業電話逐字稿") as demo:
    gr.Markdown(
        "# 台灣金融業電話逐字稿\n"
        "TEA-ASR-1.1 主稿＋Whisper large-v2 比對稿；Qwen3-ForcedAligner 時間戳；"
        "Pyannote Community-1 說話者 A／B。兩份文字依時間逐段比較，低於 90% 的段落會標紅。"
        "點擊主逐字稿可跳到該時間播放。"
    )
    with gr.Row():
        audio = gr.Audio(label="上傳單聲道電話錄音", sources=["upload"], type="filepath", elem_id="audio_player")
        with gr.Column():
            profile = gr.Dropdown(list(PROFILES), value="平衡（建議先用）", label="錄音情境")
            device = gr.Dropdown(["自動", "cuda", "cpu"], value="自動", label="運算裝置")
            compute = gr.Dropdown(
                ["自動", "float16", "int8_float16", "int8"],
                value="自動",
                label="Whisper 計算格式",
            )
            start = gr.Button("開始產生逐字稿", variant="primary")
    with gr.Accordion("進階參數", open=False):
        with gr.Row():
            threshold = gr.Slider(0.1, 0.8, 0.35, step=0.01, label="語音偵測門檻")
            speech_ms = gr.Slider(40, 500, 100, step=10, label="最短語音（毫秒）")
            silence_ms = gr.Slider(100, 2000, 450, step=50, label="切段靜音（毫秒）")
            pad_ms = gr.Slider(0, 600, 250, step=25, label="前後保留（毫秒）")
        with gr.Row():
            min_overlap = gr.Slider(0.05, 1.0, 0.20, step=0.05, label="最短重疊區間（秒）")
            overlap_ratio = gr.Slider(0.05, 0.9, 0.25, step=0.05, label="字詞重疊比例")
        prompt = gr.Textbox(PROMPT, label="金融領域提示（兩個模型共用）", lines=3)
        hotwords = gr.Textbox(HOTWORDS, label="金融關鍵詞", lines=2)
    with gr.Accordion("離線模型路徑", open=False):
        whisper_path = gr.Textbox(str(WHISPER_DIR), label="Whisper large-v2 模型資料夾")
        tea_path = gr.Textbox(str(TEA_DIR), label="TEA-ASR-1.1 模型資料夾")
        aligner_path = gr.Textbox(str(ALIGNER_DIR), label="Qwen3-ForcedAligner 模型資料夾")
        pyannote_path = gr.Textbox(str(PYANNOTE_DIR), label="Pyannote Community-1 模型資料夾")
    status = gr.Markdown("尚未開始。")
    gr.Markdown("## TEA-ASR-1.1 主逐字稿")
    transcript_html = gr.HTML('<div class="empty">完成後顯示在這裡。</div>')
    whisper_comparison = gr.Textbox(
        label="Whisper large-v2 比對稿（不會自動覆蓋主稿）",
        lines=8,
        interactive=False,
    )
    downloads = gr.File(label="下載 JSON 與 TXT", file_count="multiple")
    with gr.Accordion("原始 JSON", open=False):
        raw_json = gr.JSON()

    profile.change(apply_profile, profile, [threshold, speech_ms, silence_ms, pad_ms], api_name=False)
    start.click(
        process,
        [
            audio,
            whisper_path,
            tea_path,
            aligner_path,
            pyannote_path,
            device,
            compute,
            threshold,
            speech_ms,
            silence_ms,
            pad_ms,
            min_overlap,
            overlap_ratio,
            prompt,
            hotwords,
        ],
        [status, transcript_html, whisper_comparison, downloads, raw_json],
        concurrency_limit=1,
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
    )
