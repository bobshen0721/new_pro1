from __future__ import annotations

import unicodedata
from typing import Any

from opencc import OpenCC
from rapidfuzz.distance import Levenshtein


SIMILARITY_THRESHOLD = 0.90
T2S_CONVERTER = OpenCC("t2s")


def append_text(current: str, addition: str) -> str:
    """Join ASR units without introducing spaces between Chinese characters."""
    if not addition:
        return current
    if not current:
        return addition.lstrip()
    if addition[0].isspace() or current[-1].isspace():
        return current + addition
    left = current[-1]
    right = addition[0]
    if left.isascii() and right.isascii() and left.isalnum() and right.isalnum():
        return current + " " + addition
    return current + addition


def join_texts(parts: list[str]) -> str:
    text = ""
    for part in parts:
        text = append_text(text, part.strip())
    return text.strip()


def normalize_for_similarity(text: str) -> str:
    """Normalize script and formatting without changing the displayed transcript."""
    normalized = unicodedata.normalize("NFKC", text or "").lower()
    normalized = T2S_CONVERTER.convert(normalized)
    kept = []
    for index, char in enumerate(normalized):
        if char.isalnum():
            kept.append(char)
        elif char == "%":
            kept.append(char)
        elif (
            char in ".-/"
            and 0 < index < len(normalized) - 1
            and normalized[index - 1].isdigit()
            and normalized[index + 1].isdigit()
        ):
            kept.append(char)
    return "".join(kept)


def transcript_similarity(primary_text: str, comparison_text: str) -> float:
    primary = normalize_for_similarity(primary_text)
    comparison = normalize_for_similarity(comparison_text)
    if not primary or not comparison:
        return 0.0
    return float(Levenshtein.normalized_similarity(primary, comparison))


def _overlap_seconds(a1: float, a2: float, b1: float, b2: float) -> float:
    return max(0.0, min(a2, b2) - max(a1, b1))


def _time_gap(a1: float, a2: float, b1: float, b2: float) -> float:
    if a2 < b1:
        return b1 - a2
    if b2 < a1:
        return a1 - b2
    return 0.0


def _best_row_index(unit: dict[str, Any], rows: list[dict[str, Any]]) -> int:
    """Assign a Whisper time unit to exactly one closest TEA segment."""
    unit_start = float(unit["start"])
    unit_end = max(unit_start, float(unit["end"]))
    unit_duration = max(0.01, unit_end - unit_start)
    unit_middle = (unit_start + unit_end) / 2.0

    best_index = 0
    best_key: tuple[float, ...] | None = None
    for index, row in enumerate(rows):
        row_start = float(row["start"])
        row_end = max(row_start, float(row["end"]))
        overlap = _overlap_seconds(unit_start, unit_end, row_start, row_end)
        middle_distance = abs(unit_middle - (row_start + row_end) / 2.0)
        if overlap > 0:
            key = (0.0, -(overlap / unit_duration), middle_distance)
        else:
            key = (1.0, _time_gap(unit_start, unit_end, row_start, row_end), middle_distance)
        if best_key is None or key < best_key:
            best_key = key
            best_index = index
    return best_index


def score_time_aligned_segments(
    primary_rows: list[dict[str, Any]],
    comparison_words: list[dict[str, Any]],
    comparison_segments: list[dict[str, Any]],
    threshold: float = SIMILARITY_THRESHOLD,
) -> tuple[list[dict[str, Any]], float, str]:
    """Align Whisper units by time, score every TEA row, and return a weighted total."""
    rows = [dict(row) for row in primary_rows]
    if not rows:
        source = "whisper_words" if comparison_words else "whisper_segments"
        return [], 0.0, source

    use_words = bool(comparison_words)
    units = comparison_words if use_words else comparison_segments
    alignment_source = "whisper_words" if use_words else "whisper_segments"
    buckets: list[list[dict[str, Any]]] = [[] for _ in rows]

    for unit in sorted(units, key=lambda item: (float(item["start"]), float(item["end"]))):
        text = str(unit.get("text", "")).strip()
        if not text:
            continue
        buckets[_best_row_index(unit, rows)].append(unit)

    weighted_score = 0.0
    total_weight = 0
    for row, bucket in zip(rows, buckets):
        comparison_text = join_texts([str(item.get("text", "")) for item in bucket])
        similarity = transcript_similarity(str(row.get("text", "")), comparison_text)
        weight = max(
            len(normalize_for_similarity(str(row.get("text", "")))),
            len(normalize_for_similarity(comparison_text)),
            1,
        )
        weighted_score += similarity * weight
        total_weight += weight
        row.update(
            {
                "comparison_text": comparison_text,
                "similarity_percent": round(similarity * 100.0, 2),
                "needs_manual_review": similarity < threshold,
                "comparison_alignment_source": alignment_source,
            }
        )

    overall_similarity = weighted_score / total_weight if total_weight else 0.0
    return rows, overall_similarity, alignment_source
