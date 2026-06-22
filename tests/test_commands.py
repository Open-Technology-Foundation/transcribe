#!/usr/bin/env python3
"""
Unit tests for transcribe_pkg.cli.commands.

These tests are hermetic: all network / LLM / Whisper / audio I/O is mocked.
They target three audited defects in commands.py:
  1. -T/--timestamps with FILE output dropped per-segment timing.
  2. clean_transcript_command performed no numeric validation.
  3. Transcript files were opened without encoding="utf-8".
"""
import io
import logging
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from transcribe_pkg.cli import commands


class TestWriteOutputTimestampsToFile(unittest.TestCase):
  """Finding 1: -T with file output must emit per-segment timestamp lines."""

  def setUp(self):
    self.temp_dir = tempfile.TemporaryDirectory()
    self.output_path = os.path.join(self.temp_dir.name, "out.txt")
    self.logger = logging.getLogger("test_write_output")
    self.result = {
      "text": "hello world goodbye",
      "segments": [
        {"start": 0.0, "end": 1.5, "text": " hello world "},
        {"start": 1.5, "end": 3.0, "text": "goodbye"},
      ],
    }

  def tearDown(self):
    self.temp_dir.cleanup()

  def test_timestamps_written_to_file(self):
    """With timestamps and no subtitle format, the file gets timestamp lines."""
    ok = commands._write_output(
      result=self.result,
      output_path=self.output_path,
      use_stdout=False,
      with_timestamps=True,
      subtitle_format=None,
      logger=self.logger,
    )
    self.assertTrue(ok)
    with open(self.output_path, encoding="utf-8") as f:
      content = f.read()
    # The plain-text path would have written "hello world goodbye" with no
    # timing markers; the fix must mirror the stdout contract.
    self.assertIn("[0.00 -> 1.50] hello world", content)
    self.assertIn("[1.50 -> 3.00] goodbye", content)

  def test_no_segments_falls_back_to_plain_text(self):
    """With timestamps requested but no segments, write plain text."""
    result = {"text": "plain text only", "segments": []}
    ok = commands._write_output(
      result=result,
      output_path=self.output_path,
      use_stdout=False,
      with_timestamps=True,
      subtitle_format=None,
      logger=self.logger,
    )
    self.assertTrue(ok)
    with open(self.output_path, encoding="utf-8") as f:
      content = f.read()
    self.assertEqual(content.strip(), "plain text only")
    self.assertNotIn("->", content)


class TestCleanTranscriptValidation(unittest.TestCase):
  """Finding 2: clean_transcript_command must validate numeric args."""

  def setUp(self):
    self.temp_dir = tempfile.TemporaryDirectory()
    self.input_path = os.path.join(self.temp_dir.name, "raw.txt")
    with open(self.input_path, "w", encoding="utf-8") as f:
      f.write("some transcript text")

  def tearDown(self):
    self.temp_dir.cleanup()

  @patch("transcribe_pkg.core.processor.TranscriptProcessor")
  def test_max_chunk_size_zero_rejected(self, mock_processor_cls):
    """-s 0 must be rejected before constructing the processor."""
    rc = commands.clean_transcript_command([self.input_path, "-s", "0"])
    self.assertEqual(rc, 1)
    mock_processor_cls.assert_not_called()

  @patch("transcribe_pkg.core.processor.TranscriptProcessor")
  def test_temperature_out_of_range_rejected(self, mock_processor_cls):
    """-t 9 (outside 0.0-1.0) must be rejected."""
    rc = commands.clean_transcript_command([self.input_path, "-t", "9"])
    self.assertEqual(rc, 1)
    mock_processor_cls.assert_not_called()

  @patch("transcribe_pkg.core.processor.TranscriptProcessor")
  def test_max_tokens_negative_rejected(self, mock_processor_cls):
    """-M -1 must be rejected."""
    rc = commands.clean_transcript_command([self.input_path, "-M", "-1"])
    self.assertEqual(rc, 1)
    mock_processor_cls.assert_not_called()

  @patch("transcribe_pkg.core.processor.TranscriptProcessor")
  def test_valid_args_proceed(self, mock_processor_cls):
    """Valid numeric args must NOT be rejected; processor is used."""
    mock_proc = MagicMock()
    mock_proc.process.return_value = "cleaned"
    mock_processor_cls.return_value = mock_proc
    rc = commands.clean_transcript_command([self.input_path, "-s", "100",
                                            "-t", "0.2", "-M", "1000"])
    self.assertEqual(rc, 0)
    mock_processor_cls.assert_called_once()


class TestUtf8Encoding(unittest.TestCase):
  """Finding 3: transcript files must be opened with encoding='utf-8'."""

  def setUp(self):
    self.logger = logging.getLogger("test_utf8")

  def test_write_output_uses_utf8(self):
    """_write_output plain-text path opens the file with encoding='utf-8'."""
    opened = {}

    def tracking_open(path, mode="r", *args, **kwargs):
      opened["mode"] = mode
      opened["encoding"] = kwargs.get("encoding")
      return io.StringIO()

    with patch("builtins.open", side_effect=tracking_open):
      commands._write_output(
        result="Naïve café résumé — ünïcödé",
        output_path="/tmp/does-not-matter.txt",
        use_stdout=False,
        with_timestamps=False,
        subtitle_format=None,
        logger=self.logger,
      )
    self.assertEqual(opened.get("encoding"), "utf-8")

  def test_save_raw_transcript_uses_utf8(self):
    """_save_raw_transcript opens the .raw file with encoding='utf-8'."""
    opened = {}

    def tracking_open(path, mode="r", *args, **kwargs):
      opened["encoding"] = kwargs.get("encoding")
      return io.StringIO()

    parsed = MagicMock()
    parsed.raw = True
    with patch("builtins.open", side_effect=tracking_open):
      commands._save_raw_transcript(
        result="café",
        output_path="/tmp/out.txt",
        parsed_args=parsed,
        with_timestamps=False,
        logger=self.logger,
      )
    self.assertEqual(opened.get("encoding"), "utf-8")

  def test_clean_transcript_read_and_write_use_utf8(self):
    """clean_transcript_command reads input and writes output as utf-8."""
    temp_dir = tempfile.TemporaryDirectory()
    self.addCleanup(temp_dir.cleanup)
    input_path = os.path.join(temp_dir.name, "in.txt")
    output_path = os.path.join(temp_dir.name, "out.txt")
    # Non-ASCII content round-trips cleanly under utf-8.
    with open(input_path, "w", encoding="utf-8") as f:
      f.write("café résumé")

    with patch("transcribe_pkg.core.processor.TranscriptProcessor") as mock_cls:
      mock_proc = MagicMock()
      mock_proc.process.return_value = "café cleaned ünïcödé"
      mock_cls.return_value = mock_proc
      rc = commands.clean_transcript_command([input_path, "-o", output_path])

    self.assertEqual(rc, 0)
    with open(output_path, encoding="utf-8") as f:
      self.assertEqual(f.read(), "café cleaned ünïcödé")


if __name__ == "__main__":
  unittest.main()

#fin
