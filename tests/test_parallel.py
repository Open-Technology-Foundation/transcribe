#!/usr/bin/env python3
"""
Tests for the parallel processing module.

These tests exercise ParallelProcessor in isolation. All process functions
are local fakes (no network, no LLM, no Whisper, no audio decode), so the
tests are hermetic, fast and deterministic.
"""
import logging
import unittest
from unittest.mock import MagicMock, patch

from transcribe_pkg.core.parallel import ParallelProcessor
from transcribe_pkg.utils.api_utils import APIError


def _upper(chunk, kwargs):
  """Trivial process_func: uppercase the chunk."""
  return chunk.upper()


def _boom(chunk, kwargs):
  """process_func that always fails."""
  raise ValueError("worker exploded")


class TestProcessTextFailureAggregation(unittest.TestCase):
  """Finding (1): worker failures must surface an aggregate signal."""

  def _make(self, **kw):
    return ParallelProcessor(max_workers=2, logger=MagicMock(spec=logging.Logger), **kw)

  @patch("transcribe_pkg.core.parallel.split_text_for_processing")
  def test_all_chunks_fail_raises(self, mock_split):
    """If every chunk fails, process_text must raise rather than return raw text."""
    mock_split.return_value = ["alpha", "beta", "gamma"]
    proc = self._make()
    with self.assertRaises(APIError):
      proc.process_text("ignored", _boom, show_progress=False)

  @patch("transcribe_pkg.core.parallel.split_text_for_processing")
  def test_all_chunks_fail_does_not_return_raw(self, mock_split):
    """The silent-success bug: raw input must not be returned on total failure."""
    mock_split.return_value = ["alpha", "beta"]
    proc = self._make()
    try:
      result = proc.process_text("ignored", _boom, show_progress=False)
    except APIError:
      return  # raising is the correct behavior
    self.fail(f"Expected APIError on total failure, got result: {result!r}")

  @patch("transcribe_pkg.core.parallel.split_text_for_processing")
  def test_partial_failure_warns_and_falls_back(self, mock_split):
    """Partial failure: succeed overall, substitute raw text, and warn."""
    mock_split.return_value = ["good", "bad"]

    def _func(chunk, kwargs):
      if chunk == "bad":
        raise ValueError("nope")
      return chunk.upper()

    proc = self._make()
    result = proc.process_text("ignored", _func, show_progress=False)
    # Successful chunk processed, failed chunk fell back to raw text.
    self.assertIn("GOOD", result)
    self.assertIn("bad", result)
    # A warning about the fallback must have been logged.
    self.assertTrue(
      proc.logger.warning.called,
      "Expected a warning when some chunks fall back to raw text",
    )

  @patch("transcribe_pkg.core.parallel.split_text_for_processing")
  def test_all_success_no_warning(self, mock_split):
    """All chunks succeed: no warning, no raise, processed output returned."""
    mock_split.return_value = ["a", "b", "c"]
    proc = self._make()
    result = proc.process_text("ignored", _upper, show_progress=False)
    self.assertEqual(result, "A\n\nB\n\nC")
    self.assertFalse(proc.logger.warning.called)


class TestProcessTextChunkIndex(unittest.TestCase):
  """Finding (2): process_text must inject a distinct chunk_index per chunk."""

  def _make(self, **kw):
    return ParallelProcessor(max_workers=4, logger=MagicMock(spec=logging.Logger), **kw)

  @patch("transcribe_pkg.core.parallel.split_text_for_processing")
  def test_each_chunk_gets_its_own_index(self, mock_split):
    """Every worker must see chunk_index matching its position, not 0."""
    mock_split.return_value = ["a", "b", "c", "d"]
    seen = []

    def _func(chunk, kwargs):
      seen.append(kwargs.get("chunk_index", 0))
      return chunk.upper()

    proc = self._make()
    proc.process_text("ignored", _func, show_progress=False)
    self.assertEqual(set(seen), {0, 1, 2, 3})

  @patch("transcribe_pkg.core.parallel.split_text_for_processing")
  def test_caller_kwargs_not_mutated(self, mock_split):
    """Injecting chunk_index must not leak back into the caller's kwargs dict."""
    mock_split.return_value = ["a", "b"]

    def _func(chunk, kwargs):
      return chunk.upper()

    proc = self._make()
    proc.process_text("ignored", _func, show_progress=False, context="ctx")
    # process_text receives caller kwargs via **kwargs; the per-chunk copy
    # must carry chunk_index without contaminating sibling chunks. Verify the
    # extra kwarg still reaches the worker and chunk_index is per-chunk.
    captured = {}

    def _capture(chunk, kwargs):
      captured[chunk] = dict(kwargs)
      return chunk

    proc.process_text("ignored", _capture, show_progress=False, context="ctx")
    self.assertEqual(captured["a"]["chunk_index"], 0)
    self.assertEqual(captured["b"]["chunk_index"], 1)
    self.assertEqual(captured["a"]["context"], "ctx")
    self.assertEqual(captured["b"]["context"], "ctx")


class TestUseProcessesFallback(unittest.TestCase):
  """Finding (3): use_processes=True must fall back to threads, not crash."""

  def test_process_pool_request_warns(self):
    """Requesting a process pool logs a warning about it being unsupported."""
    logger = MagicMock(spec=logging.Logger)
    ParallelProcessor(max_workers=2, use_processes=True, logger=logger)
    self.assertTrue(logger.warning.called, "Expected a warning for use_processes=True")
    warned = " ".join(str(c.args[0]) for c in logger.warning.call_args_list).lower()
    self.assertIn("process", warned)

  def test_public_parameter_still_accepted(self):
    """The public use_processes ctor parameter must remain accepted."""
    # Must not raise; constructing with the option is part of the public API.
    proc = ParallelProcessor(max_workers=2, use_processes=True,
                             logger=MagicMock(spec=logging.Logger))
    self.assertIsInstance(proc, ParallelProcessor)

  @patch("transcribe_pkg.core.parallel.split_text_for_processing")
  def test_process_text_with_unpicklable_closure_does_not_crash(self, mock_split):
    """With use_processes=True, an unpicklable closure must still run on threads."""
    mock_split.return_value = ["x", "y", "z"]
    captured_state = {"suffix": "!"}

    def closure_func(chunk, kwargs):
      # Captures captured_state -> unpicklable local closure; only a thread
      # pool can run this without a PicklingError.
      return chunk.upper() + captured_state["suffix"]

    proc = ParallelProcessor(max_workers=2, use_processes=True,
                             logger=MagicMock(spec=logging.Logger))
    result = proc.process_text("ignored", closure_func, show_progress=False)
    self.assertEqual(result, "X!\n\nY!\n\nZ!")

  @patch("transcribe_pkg.core.parallel.split_text_for_processing")
  def test_process_audio_chunks_with_closure_does_not_crash(self, mock_split):
    """process_audio_chunks must also fall back to threads for closures."""
    captured_state = {"n": 7}

    def closure_func(path, kwargs):
      return f"{path}:{captured_state['n']}"

    proc = ParallelProcessor(max_workers=2, use_processes=True,
                             logger=MagicMock(spec=logging.Logger))
    result = proc.process_audio_chunks(["p0", "p1"], closure_func, show_progress=False)
    self.assertEqual(result, ["p0:7", "p1:7"])


class TestProcessAudioChunksPositions(unittest.TestCase):
  """Finding (4): failed audio chunks must preserve positional correspondence."""

  def _make(self, **kw):
    return ParallelProcessor(max_workers=3, logger=MagicMock(spec=logging.Logger), **kw)

  def test_failed_chunk_keeps_position(self):
    """A failed middle chunk must leave a placeholder, not shift later chunks."""
    def _func(path, kwargs):
      if path == "p1":
        raise ValueError("decode failed")
      return f"text-{path}"

    proc = self._make()
    result = proc.process_audio_chunks(["p0", "p1", "p2"], _func, show_progress=False)
    # Positional correspondence preserved: index 1 is the failure placeholder.
    self.assertEqual(len(result), 3)
    self.assertEqual(result[0], "text-p0")
    self.assertIsNone(result[1])
    self.assertEqual(result[2], "text-p2")

  def test_trailing_failure_keeps_length(self):
    """A failed last chunk must not silently shorten the result list."""
    def _func(path, kwargs):
      if path == "p2":
        raise ValueError("boom")
      return f"t-{path}"

    proc = self._make()
    result = proc.process_audio_chunks(["p0", "p1", "p2"], _func, show_progress=False)
    self.assertEqual(len(result), 3)
    self.assertIsNone(result[2])

  def test_missing_index_logged(self):
    """The missing chunk index must be logged for visibility."""
    def _func(path, kwargs):
      if path == "p0":
        raise ValueError("boom")
      return path

    proc = self._make()
    proc.process_audio_chunks(["p0", "p1"], _func, show_progress=False)
    logged = " ".join(
      str(c.args[0]) for c in proc.logger.warning.call_args_list
    ) + " " + " ".join(
      str(c.args[0]) for c in proc.logger.error.call_args_list
    )
    self.assertIn("0", logged)

  def test_all_success_positions(self):
    """Happy path: all chunks present in order."""
    def _func(path, kwargs):
      return f"t-{path}"

    proc = self._make()
    result = proc.process_audio_chunks(["p0", "p1", "p2"], _func, show_progress=False)
    self.assertEqual(result, ["t-p0", "t-p1", "t-p2"])


if __name__ == "__main__":
  unittest.main()

#fin
