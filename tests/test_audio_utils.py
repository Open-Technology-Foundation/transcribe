#!/usr/bin/env python3
"""
Tests for audio utilities (AudioProcessor).

These tests are hermetic: pydub decoding and AudioSegment export are mocked,
so no real audio files are decoded and no external tools are invoked.
"""
import os
import unittest
from unittest.mock import patch, MagicMock

from transcribe_pkg.utils.audio_utils import AudioProcessor


def _fake_audio(length_ms):
    """
    Build a fake AudioSegment-like mock.

    Supports len(audio) -> length_ms and audio[i:j] slicing returning a
    chunk mock whose .export() is a harmless no-op.
    """
    chunk = MagicMock(name="AudioChunk")
    chunk.export = MagicMock(return_value=None)

    audio = MagicMock(name="AudioSegment")
    audio.__len__ = MagicMock(return_value=length_ms)
    audio.__getitem__ = MagicMock(return_value=chunk)
    return audio


class TestSplitAudioZeroDuration(unittest.TestCase):
    """Finding (1): zero-duration decodable audio must raise, not silently
    return an empty list and leak a temp dir."""

    def test_zero_length_audio_raises_value_error(self):
        processor = AudioProcessor()
        with patch.object(processor, "load_audio", return_value=_fake_audio(0)):
            with self.assertRaises(ValueError) as ctx:
                processor.split_audio("/path/to/silent.mp3")
        self.assertIn("no audio data", str(ctx.exception))

    def test_zero_length_audio_does_not_leak_temp_dir(self):
        """No temp directory should be created when the guard fires."""
        processor = AudioProcessor()
        created_dirs = []
        real_mkdtemp = __import__("tempfile").mkdtemp

        def tracking_mkdtemp(*args, **kwargs):
            d = real_mkdtemp(*args, **kwargs)
            created_dirs.append(d)
            return d

        with patch.object(processor, "load_audio", return_value=_fake_audio(0)):
            with patch("transcribe_pkg.utils.audio_utils.tempfile.mkdtemp",
                       side_effect=tracking_mkdtemp):
                with self.assertRaises(ValueError):
                    processor.split_audio("/path/to/silent.mp3")
        # Either no temp dir was created, or any created dir was cleaned up.
        for d in created_dirs:
            self.assertFalse(os.path.exists(d),
                             f"Temp dir leaked on zero-length audio: {d}")


class TestSplitAudioTempDirCleanup(unittest.TestCase):
    """Finding (2): cleanup() must remove the temp directory, not just the
    chunk files inside it."""

    def test_cleanup_removes_temp_directory(self):
        processor = AudioProcessor()
        # 3 chunks of 10s out of a 25s clip -> non-empty, real temp dir created.
        with patch.object(processor, "load_audio",
                          return_value=_fake_audio(25000)):
            chunks = processor.split_audio("/path/to/audio.mp3",
                                           chunk_length_ms=10000)

        self.assertEqual(len(chunks), 3)
        temp_dir = os.path.dirname(chunks[0])
        self.assertTrue(os.path.isdir(temp_dir),
                        "Temp dir should exist before cleanup")

        processor.cleanup()

        self.assertFalse(os.path.exists(temp_dir),
                         f"cleanup() leaked temp dir: {temp_dir}")

    def test_cleanup_is_idempotent(self):
        """Calling cleanup() twice must not raise even after dir removal."""
        processor = AudioProcessor()
        with patch.object(processor, "load_audio",
                          return_value=_fake_audio(15000)):
            chunks = processor.split_audio("/path/to/audio.mp3",
                                           chunk_length_ms=10000)
        temp_dir = os.path.dirname(chunks[0])
        processor.cleanup()
        # Second call should be a harmless no-op.
        processor.cleanup()
        self.assertFalse(os.path.exists(temp_dir))


if __name__ == "__main__":
    unittest.main()

#fin
