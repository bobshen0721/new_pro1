import unittest

from comparison import normalize_for_similarity, score_time_aligned_segments


def primary_row(start, end, text):
    return {
        "start": start,
        "end": end,
        "text": text,
        "speaker": "A",
        "suspected_overlap": False,
    }


class SegmentSimilarityTests(unittest.TestCase):
    def test_normalization_keeps_financial_numbers(self):
        self.assertEqual(normalize_for_similarity("申購：５００萬，報酬 3.5％"), "申购500万报酬3.5%")

    def test_scores_each_time_aligned_segment(self):
        rows = [
            primary_row(0.0, 2.0, "您好"),
            primary_row(2.0, 4.0, "申購500萬"),
        ]
        words = [
            {"start": 0.1, "end": 0.8, "text": "您好"},
            {"start": 2.1, "end": 2.8, "text": "申購"},
            {"start": 2.8, "end": 3.5, "text": "50萬"},
        ]

        scored, overall, source = score_time_aligned_segments(rows, words, [])

        self.assertEqual(source, "whisper_words")
        self.assertEqual(scored[0]["similarity_percent"], 100.0)
        self.assertFalse(scored[0]["needs_manual_review"])
        self.assertEqual(scored[1]["comparison_text"], "申購50萬")
        self.assertAlmostEqual(scored[1]["similarity_percent"], 83.33, places=2)
        self.assertTrue(scored[1]["needs_manual_review"])
        self.assertAlmostEqual(overall * 100.0, 87.5, places=2)

    def test_missing_whisper_text_is_flagged(self):
        scored, overall, source = score_time_aligned_segments(
            [primary_row(0.0, 2.0, "請確認身分")],
            [],
            [],
        )

        self.assertEqual(source, "whisper_segments")
        self.assertEqual(overall, 0.0)
        self.assertEqual(scored[0]["comparison_text"], "")
        self.assertEqual(scored[0]["similarity_percent"], 0.0)
        self.assertTrue(scored[0]["needs_manual_review"])

    def test_segment_timestamps_are_used_when_words_are_unavailable(self):
        scored, _, source = score_time_aligned_segments(
            [primary_row(5.0, 8.0, "風險報酬等級三")],
            [],
            [{"start": 5.1, "end": 7.9, "text": "風險報酬等級三"}],
        )

        self.assertEqual(source, "whisper_segments")
        self.assertEqual(scored[0]["similarity_percent"], 100.0)
        self.assertFalse(scored[0]["needs_manual_review"])


if __name__ == "__main__":
    unittest.main()
