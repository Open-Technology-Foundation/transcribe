#!/usr/bin/env python3
"""
Tests for the core Transcriber timestamp-stitching and failure-signalling logic.

These tests are hermetic: the OpenAIClient and AudioProcessor are replaced with
MagicMock instances injected through the Transcriber constructor, so no network,
no API key, and no real audio decoding occur.
"""
import itertools
import logging
import unittest
from unittest.mock import MagicMock

from transcribe_pkg.core.transcriber import Transcriber

# Monotonic counter so each test gets a private, freshly-named logger. Sharing a
# logger by name across tests would leak handler/level/monkeypatch state between
# them (e.g. one test replacing ``logger.warning`` would silence assertLogs in
# another).
_logger_counter = itertools.count()


class _Segment:
  """Minimal stand-in for an OpenAI verbose_json segment object.

  The production code branches on ``hasattr(segment, 'id')`` to treat the
  segment as an object (vs a dict), so exposing ``id`` is what selects the
  object code path.
  """

  def __init__(self, start, end, text="seg", seg_id=0, words=None):
    self.id = seg_id
    self.start = start
    self.end = end
    self.text = text
    if words is not None:
      self.words = words


class _Verbose:
  """Minimal stand-in for a verbose_json transcription response object."""

  def __init__(self, text, segments):
    self.text = text
    self.segments = segments


def _make_transcriber(chunk_length_ms=1000):
  """Build a Transcriber with mocked audio + API clients and a private logger."""
  logger = logging.getLogger(f"test_transcriber.{next(_logger_counter)}")
  logger.handlers = [logging.NullHandler()]
  logger.propagate = False
  logger.setLevel(logging.CRITICAL)  # silence + keep tqdm progress bars disabled
  api_client = MagicMock()
  audio_processor = MagicMock()
  transcriber = Transcriber(
    api_client=api_client,
    audio_processor=audio_processor,
    chunk_length_ms=chunk_length_ms,
    logger=logger,
  )
  return transcriber, api_client, audio_processor


class TestTimestampDrift(unittest.TestCase):
  """Finding 1: cumulative timestamp drift across stitched chunks."""

  def test_sequential_offset_is_fixed_per_index_not_accumulated(self):
    # Each chunk is 1000 ms (1.0 s) of audio, but the last speech segment in
    # each chunk ends at 0.5 s (trailing silence). The correct global offset
    # for chunk i is i * 1.0 s, independent of where speech stops.
    transcriber, api_client, _ = _make_transcriber(chunk_length_ms=1000)
    api_client.transcribe_audio.side_effect = [
      _Verbose("chunk0", [_Segment(0.0, 0.5, "c0")]),
      _Verbose("chunk1", [_Segment(0.0, 0.5, "c1")]),
      _Verbose("chunk2", [_Segment(0.0, 0.5, "c2")]),
    ]

    result = transcriber._transcribe_chunks_sequential(
      ["c0.wav", "c1.wav", "c2.wav"], with_timestamps=True
    )

    starts = [seg["start"] for seg in result["segments"]]
    # Chunk i's segment must start at i * 1.0 s. The buggy accumulator advanced
    # by 0.5 s per chunk, yielding [0.0, 0.5, 1.0].
    self.assertEqual(starts, [0.0, 1.0, 2.0])

  def test_parallel_offset_is_fixed_per_index_not_accumulated(self):
    transcriber, api_client, _ = _make_transcriber(chunk_length_ms=1000)
    api_client.transcribe_audio.side_effect = [
      _Verbose("chunk0", [_Segment(0.0, 0.5, "c0")]),
      _Verbose("chunk1", [_Segment(0.0, 0.5, "c1")]),
      _Verbose("chunk2", [_Segment(0.0, 0.5, "c2")]),
    ]

    result = transcriber._transcribe_chunks_parallel(
      ["c0.wav", "c1.wav", "c2.wav"], with_timestamps=True, max_workers=3
    )

    starts = [seg["start"] for seg in result["segments"]]
    self.assertEqual(starts, [0.0, 1.0, 2.0])

  def test_sequential_word_timestamps_use_fixed_offset(self):
    # Word-level timestamps must use the same fixed per-chunk offset.
    transcriber, api_client, _ = _make_transcriber(chunk_length_ms=1000)
    word0 = MagicMock(spec=["word", "start", "end"])
    word0.word, word0.start, word0.end = "hi", 0.1, 0.4
    word1 = MagicMock(spec=["word", "start", "end"])
    word1.word, word1.start, word1.end = "yo", 0.1, 0.4
    api_client.transcribe_audio.side_effect = [
      _Verbose("c0", [_Segment(0.0, 0.5, "c0", words=[word0])]),
      _Verbose("c1", [_Segment(0.0, 0.5, "c1", words=[word1])]),
    ]

    result = transcriber._transcribe_chunks_sequential(
      ["c0.wav", "c1.wav"], with_timestamps=True
    )

    word_starts = [seg["words"][0]["start"] for seg in result["segments"]]
    self.assertEqual(word_starts, [0.1, 1.1])

  def test_sequential_dict_response_uses_fixed_offset(self):
    # Whisper verbose_json may arrive as plain dicts; the dict code path must
    # also apply the fixed per-chunk offset rather than accumulating.
    transcriber, api_client, _ = _make_transcriber(chunk_length_ms=1000)
    api_client.transcribe_audio.side_effect = [
      {"text": "c0", "segments": [{"id": 0, "start": 0.0, "end": 0.5, "text": "c0"}]},
      {"text": "c1", "segments": [{"id": 0, "start": 0.0, "end": 0.5, "text": "c1"}]},
    ]

    result = transcriber._transcribe_chunks_sequential(
      ["c0.wav", "c1.wav"], with_timestamps=True
    )

    starts = [seg["start"] for seg in result["segments"]]
    self.assertEqual(starts, [0.0, 1.0])


class TestSkippedChunkOffset(unittest.TestCase):
  """Finding 2: a skipped/failed chunk must still advance the offset."""

  def test_sequential_empty_chunk_still_advances_offset(self):
    # Chunk 0 yields an empty transcription (hits `continue`); chunk 1's
    # segment must still be offset by one full chunk width (1.0 s).
    transcriber, api_client, _ = _make_transcriber(chunk_length_ms=1000)
    api_client.transcribe_audio.side_effect = [
      _Verbose("", []),  # empty text -> skipped
      _Verbose("chunk1", [_Segment(0.0, 0.5, "c1")]),
    ]

    result = transcriber._transcribe_chunks_sequential(
      ["c0.wav", "c1.wav"], with_timestamps=True
    )

    self.assertEqual(len(result["segments"]), 1)
    self.assertEqual(result["segments"][0]["start"], 1.0)
    self.assertEqual(result["segments"][0]["end"], 1.5)

  def test_parallel_failed_chunk_still_advances_offset(self):
    # Chunk 0 failed in its worker (-> {"text": "", "segments": []}); chunk 1
    # must still be offset by one full chunk width.
    transcriber, api_client, _ = _make_transcriber(chunk_length_ms=1000)
    api_client.transcribe_audio.side_effect = [
      Exception("boom on chunk 0"),
      _Verbose("chunk1", [_Segment(0.0, 0.5, "c1")]),
    ]

    result = transcriber._transcribe_chunks_parallel(
      ["c0.wav", "c1.wav"], with_timestamps=True, max_workers=2
    )

    self.assertEqual(len(result["segments"]), 1)
    self.assertEqual(result["segments"][0]["start"], 1.0)
    self.assertEqual(result["segments"][0]["end"], 1.5)


class TestFailedChunkWarning(unittest.TestCase):
  """Finding 3: a non-zero failed-chunk count must emit an aggregate warning."""

  def test_sequential_failed_chunk_logs_aggregate_warning(self):
    transcriber, api_client, _ = _make_transcriber(chunk_length_ms=1000)
    api_client.transcribe_audio.side_effect = [
      "good text",
      Exception("transcription failed"),
    ]

    with self.assertLogs(transcriber.logger, level="WARNING") as cm:
      result = transcriber._transcribe_chunks_sequential(
        ["c0.wav", "c1.wav"], with_timestamps=False
      )

    self.assertEqual(result, ["good text", ""])
    self.assertTrue(
      any("1/2 chunks failed transcription" in m for m in cm.output),
      f"expected aggregate failure warning, got: {cm.output}",
    )

  def test_parallel_failed_chunk_logs_aggregate_warning(self):
    transcriber, api_client, _ = _make_transcriber(chunk_length_ms=1000)
    api_client.transcribe_audio.side_effect = [
      "good text",
      Exception("transcription failed"),
    ]

    with self.assertLogs(transcriber.logger, level="WARNING") as cm:
      result = transcriber._transcribe_chunks_parallel(
        ["c0.wav", "c1.wav"], with_timestamps=False, max_workers=2
      )

    self.assertEqual(result, ["good text", ""])
    self.assertTrue(
      any("1/2 chunks failed transcription" in m for m in cm.output),
      f"expected aggregate failure warning, got: {cm.output}",
    )

  def test_no_warning_when_all_chunks_succeed(self):
    transcriber, api_client, _ = _make_transcriber(chunk_length_ms=1000)
    api_client.transcribe_audio.side_effect = ["a", "b"]

    logger = transcriber.logger
    logger.warning = MagicMock()
    transcriber._transcribe_chunks_sequential(
      ["c0.wav", "c1.wav"], with_timestamps=False
    )

    failure_warnings = [
      call for call in logger.warning.call_args_list
      if "chunks failed transcription" in str(call)
    ]
    self.assertEqual(failure_warnings, [])


if __name__ == "__main__":
  unittest.main()

#fin
