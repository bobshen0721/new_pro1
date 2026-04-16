"""
WhisperX 語音轉文字工具
支援：逐字時間戳、說話者識別、多語言、多種輸出格式
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def load_config(config_path: str = "config.json") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def transcribe(
    audio_path: str,
    model_name: str = "large-v2",
    language: str = None,
    device: str = "cuda",
    compute_type: str = "float16",
    batch_size: int = 16,
    diarize: bool = False,
    hf_token: str = None,
    min_speakers: int = None,
    max_speakers: int = None,
    output_dir: str = "output",
    output_format: str = "all",
) -> dict:
    """
    執行 WhisperX 語音轉文字。

    Args:
        audio_path:    音訊檔路徑
        model_name:    Whisper 模型名稱 (tiny/base/small/medium/large-v2/large-v3)
        language:      語言代碼，例如 'zh'、'en'；None 表示自動偵測
        device:        運算裝置 ('cuda' 或 'cpu')
        compute_type:  精度 ('float16'、'int8'、'float32')
        batch_size:    批次大小（GPU 記憶體不足時可調低）
        diarize:       是否啟用說話者識別
        hf_token:      HuggingFace Access Token（diarize 必須）
        min_speakers:  最少說話者人數
        max_speakers:  最多說話者人數
        output_dir:    輸出資料夾
        output_format: 輸出格式 ('srt'、'vtt'、'txt'、'json'、'all')

    Returns:
        包含轉錄結果的字典
    """
    try:
        import whisperx
    except ImportError:
        sys.exit("請先安裝 whisperx：pip install whisperx")

    audio_path = Path(audio_path)
    if not audio_path.exists():
        sys.exit(f"找不到音訊檔：{audio_path}")

    os.makedirs(output_dir, exist_ok=True)
    stem = audio_path.stem

    # ── 1. 載入模型並轉錄 ──────────────────────────────────────────────
    print(f"[1/4] 載入 Whisper 模型：{model_name}  裝置：{device}")
    model = whisperx.load_model(model_name, device, compute_type=compute_type)

    print(f"[2/4] 讀取音訊：{audio_path.name}")
    audio = whisperx.load_audio(str(audio_path))

    print("[2/4] 轉錄中…")
    result = model.transcribe(audio, batch_size=batch_size, language=language)
    detected_lang = result.get("language", language or "unknown")
    print(f"      偵測語言：{detected_lang}")

    # ── 2. 對齊（取得逐字時間戳）──────────────────────────────────────
    print("[3/4] 對齊時間戳…")
    try:
        align_model, metadata = whisperx.load_align_model(
            language_code=detected_lang, device=device
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
    except Exception as e:
        print(f"      警告：對齊失敗（{e}），使用原始時間戳繼續。")

    # ── 3. 說話者識別（可選）──────────────────────────────────────────
    if diarize:
        if not hf_token:
            print("      警告：未提供 HuggingFace Token，跳過說話者識別。")
        else:
            print("[4/4] 說話者識別中…")
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=hf_token, device=device
            )
            diarize_kwargs = {}
            if min_speakers:
                diarize_kwargs["min_speakers"] = min_speakers
            if max_speakers:
                diarize_kwargs["max_speakers"] = max_speakers
            diarize_segments = diarize_model(audio, **diarize_kwargs)
            result = whisperx.assign_word_speakers(diarize_segments, result)
    else:
        print("[4/4] 跳過說話者識別（未啟用）")

    # ── 4. 輸出 ───────────────────────────────────────────────────────
    _write_outputs(result, stem, output_dir, output_format, detected_lang)

    print(f"\n完成！輸出目錄：{output_dir}/")
    return result


# ── 輸出格式 ──────────────────────────────────────────────────────────────

def _format_timestamp(seconds: float, fmt: str = "srt") -> str:
    """將秒數轉換為 SRT 或 VTT 時間格式。"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    sep = "," if fmt == "srt" else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _write_outputs(result: dict, stem: str, output_dir: str, output_format: str, language: str):
    segments = result.get("segments", [])

    formats = (
        ["srt", "vtt", "txt", "json"]
        if output_format == "all"
        else [output_format]
    )

    for fmt in formats:
        out_path = Path(output_dir) / f"{stem}.{fmt}"
        if fmt == "txt":
            _write_txt(segments, out_path)
        elif fmt == "srt":
            _write_srt(segments, out_path)
        elif fmt == "vtt":
            _write_vtt(segments, out_path)
        elif fmt == "json":
            _write_json(result, out_path)
        print(f"      已儲存：{out_path}")


def _write_txt(segments: list, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            speaker = f"[{seg['speaker']}] " if "speaker" in seg else ""
            f.write(f"{speaker}{seg['text'].strip()}\n")


def _write_srt(segments: list, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = _format_timestamp(seg["start"], "srt")
            end = _format_timestamp(seg["end"], "srt")
            speaker = f"[{seg['speaker']}] " if "speaker" in seg else ""
            f.write(f"{i}\n{start} --> {end}\n{speaker}{seg['text'].strip()}\n\n")


def _write_vtt(segments: list, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, seg in enumerate(segments, 1):
            start = _format_timestamp(seg["start"], "vtt")
            end = _format_timestamp(seg["end"], "vtt")
            speaker = f"<v {seg['speaker']}>" if "speaker" in seg else ""
            f.write(f"{i}\n{start} --> {end}\n{speaker}{seg['text'].strip()}\n\n")


def _write_json(result: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


# ── CLI 入口 ──────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="WhisperX 語音轉文字（含逐字時間戳 & 說話者識別）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("audio", help="音訊檔路徑（wav / mp3 / mp4 / m4a …）")
    p.add_argument("--model", default="large-v2",
                   choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                   help="Whisper 模型")
    p.add_argument("--language", default=None,
                   help="語言代碼，例如 zh、en、ja；預設自動偵測")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--compute_type", default="float16",
                   choices=["float16", "int8", "float32"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--diarize", action="store_true", help="啟用說話者識別")
    p.add_argument("--hf_token", default=None,
                   help="HuggingFace Access Token（說話者識別必須）")
    p.add_argument("--min_speakers", type=int, default=None)
    p.add_argument("--max_speakers", type=int, default=None)
    p.add_argument("--output_dir", default="output", help="輸出資料夾")
    p.add_argument("--output_format", default="all",
                   choices=["srt", "vtt", "txt", "json", "all"])
    p.add_argument("--config", default="config.json",
                   help="設定檔路徑（可覆蓋預設值）")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # 讀取設定檔並以 CLI 參數覆蓋
    cfg = load_config(args.config)
    hf_token = args.hf_token or cfg.get("hf_token") or os.environ.get("HF_TOKEN")

    transcribe(
        audio_path=args.audio,
        model_name=args.model,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        batch_size=args.batch_size,
        diarize=args.diarize,
        hf_token=hf_token,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        output_dir=args.output_dir,
        output_format=args.output_format,
    )


if __name__ == "__main__":
    main()
