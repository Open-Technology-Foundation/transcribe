#!/usr/bin/env python3
"""
Tests for subtitle utilities (SRT / VTT generation and saving).

All I/O is hermetic: file writes go to a temp dir, no network / Whisper / LLM.
"""
import os
import tempfile
import unittest

from transcribe_pkg.utils.subtitle_utils import (
  format_timestamp_srt,
  format_timestamp_vtt,
  generate_srt,
  generate_vtt,
  save_subtitles,
)


class TestTimestampRounding(unittest.TestCase):
  """Finding 3: millisecond field must round, not truncate."""

  def setUp(self):
    # lru_cache on the formatters can persist stale results across edits.
    format_timestamp_srt.cache_clear()
    format_timestamp_vtt.cache_clear()

  def test_srt_rounds_milliseconds_up(self):
    # 59.9996s is 59999.6ms -> rounds to 60000ms -> 00:01:00,000.
    # Truncation would wrongly yield 00:00:59,999.
    self.assertEqual(format_timestamp_srt(59.9996), "00:01:00,000")

  def test_vtt_rounds_milliseconds_up(self):
    # Same value, VTT uses '.' separator -> 00:01:00.000.
    self.assertEqual(format_timestamp_vtt(59.9996), "00:01:00.000")

  def test_srt_simple_round(self):
    # 1.9999s -> 2000ms -> 00:00:02,000 (truncation gives 00:00:01,999).
    self.assertEqual(format_timestamp_srt(1.9999), "00:00:02,000")

  def test_vtt_simple_round(self):
    self.assertEqual(format_timestamp_vtt(1.9999), "00:00:02.000")

  def test_srt_exact_values_unchanged(self):
    # Whole/clean values must still format correctly.
    self.assertEqual(format_timestamp_srt(0.0), "00:00:00,000")
    self.assertEqual(format_timestamp_srt(3661.5), "01:01:01,500")

  def test_vtt_exact_values_unchanged(self):
    self.assertEqual(format_timestamp_vtt(0.0), "00:00:00.000")
    self.assertEqual(format_timestamp_vtt(3661.5), "01:01:01.500")


class TestLongSegmentSplitting(unittest.TestCase):
  """Finding 1: word-sparse long segments must not emit empty cues."""

  def _srt_text_lines(self, content):
    # Extract just the cue-text lines (drop indices, timestamps, blanks).
    out = []
    for line in content.split("\n"):
      s = line.strip()
      if not s:
        continue
      if "-->" in s:
        continue
      if s.isdigit():
        continue
      out.append(s)
    return out

  def _vtt_text_lines(self, content):
    out = []
    for line in content.split("\n"):
      s = line.strip()
      if not s or s == "WEBVTT":
        continue
      if "-->" in s:
        continue
      out.append(s)
    return out

  def _count_cues(self, content):
    # A cue is identified by its "-->" timestamp line.
    return sum(1 for line in content.split("\n") if "-->" in line)

  def test_srt_sparse_long_segment_no_empty_cue(self):
    # 15s segment, max_duration 5 -> num_parts would be 4, but only 2 words.
    # Buggy code: words_per_part = 2//4 = 0 -> 3 malformed empty-text cues
    # plus all text stranded on the final cue at ~11.25s.
    seg = [{"start": 0.0, "end": 15.0, "text": "hello world"}]
    content = generate_srt(seg, max_duration=5.0)
    texts = self._srt_text_lines(content)
    cues = self._count_cues(content)
    # Every emitted cue must have non-empty text: #cues == #non-empty-text-lines.
    self.assertEqual(cues, len(texts), f"empty-text cue(s) emitted: {content!r}")
    self.assertTrue(all(t for t in texts), f"empty cue text present: {content!r}")
    # All words preserved.
    joined = " ".join(texts)
    self.assertIn("hello", joined)
    self.assertIn("world", joined)
    # First cue must carry text from t=0, not be a stranded empty cue.
    first_ts = next(ln for ln in content.split("\n") if "-->" in ln)
    self.assertTrue(first_ts.startswith("00:00:00"), f"first cue not at t=0: {first_ts!r}")

  def test_vtt_sparse_long_segment_no_empty_cue(self):
    seg = [{"start": 0.0, "end": 15.0, "text": "hello world"}]
    content = generate_vtt(seg, max_duration=5.0)
    texts = self._vtt_text_lines(content)
    cues = self._count_cues(content)
    self.assertEqual(cues, len(texts), f"empty-text cue(s) emitted: {content!r}")
    self.assertTrue(all(t for t in texts), f"empty cue text present: {content!r}")
    joined = " ".join(texts)
    self.assertIn("hello", joined)
    self.assertIn("world", joined)
    first_ts = next(ln for ln in content.split("\n") if "-->" in ln)
    self.assertTrue(first_ts.startswith("00:00:00"), f"first cue not at t=0: {first_ts!r}")

  def test_srt_single_word_long_segment(self):
    # Extreme: one word over a long duration -> exactly one non-empty cue.
    seg = [{"start": 0.0, "end": 20.0, "text": "lonely"}]
    content = generate_srt(seg, max_duration=5.0)
    texts = self._srt_text_lines(content)
    self.assertEqual(texts, ["lonely"], f"unexpected cues: {content!r}")

  def test_srt_normal_long_segment_still_splits(self):
    # Word-rich long segment must still split into multiple cues with all text.
    words = " ".join(f"w{i}" for i in range(12))
    seg = [{"start": 0.0, "end": 15.0, "text": words}]
    content = generate_srt(seg, max_duration=5.0)
    cue_count = content.count("-->")
    self.assertGreater(cue_count, 1)
    texts = " ".join(self._srt_text_lines(content))
    for i in range(12):
      self.assertIn(f"w{i}", texts)


class TestSaveSubtitlesFormat(unittest.TestCase):
  """Finding 2: explicit format_type must win over the path extension."""

  def setUp(self):
    self._tmp = tempfile.mkdtemp()
    self.segments = {"segments": [{"start": 0.0, "end": 1.0, "text": "hi"}]}

  def _read(self, path):
    with open(path, "r", encoding="utf-8") as f:
      return f.read()

  def test_explicit_vtt_overrides_srt_extension(self):
    # format_type='vtt' but path ends .srt -> content MUST be VTT.
    out = os.path.join(self._tmp, "out.srt")
    result = save_subtitles(self.segments, out, format_type="vtt")
    self.assertIsNotNone(result)
    content = self._read(result)
    self.assertTrue(content.startswith("WEBVTT"), f"expected VTT, got: {content!r}")

  def test_explicit_srt_overrides_vtt_extension(self):
    out = os.path.join(self._tmp, "out.vtt")
    result = save_subtitles(self.segments, out, format_type="srt")
    self.assertIsNotNone(result)
    content = self._read(result)
    self.assertFalse(content.startswith("WEBVTT"), f"expected SRT, got: {content!r}")
    # SRT begins with the subtitle index "1".
    self.assertTrue(content.lstrip().startswith("1"), f"expected SRT index, got: {content!r}")

  def test_unknown_format_falls_back_to_extension(self):
    # No/invalid format_type -> derive from extension.
    out = os.path.join(self._tmp, "out.vtt")
    result = save_subtitles(self.segments, out, format_type="")
    self.assertIsNotNone(result)
    content = self._read(result)
    self.assertTrue(content.startswith("WEBVTT"))

  def test_no_segments_returns_none(self):
    out = os.path.join(self._tmp, "empty.srt")
    result = save_subtitles({"segments": []}, out, format_type="srt")
    self.assertIsNone(result)


if __name__ == "__main__":
  unittest.main()

#fin
