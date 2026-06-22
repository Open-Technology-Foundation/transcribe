#!/usr/bin/env python3
"""
Local Whisper client using faster-whisper for GPU-accelerated transcription.

This module provides a drop-in replacement for OpenAIClient.transcribe_audio()
using faster-whisper for local transcription without API costs.
"""
import logging
import os
import sys
from pathlib import Path
from typing import Any, BinaryIO, TYPE_CHECKING

from transcribe_pkg.utils.logging_utils import get_logger

if TYPE_CHECKING:
  from faster_whisper import WhisperModel


class LocalWhisperClient:
  """Drop-in replacement for OpenAIClient.transcribe_audio() using faster-whisper."""

  def __init__(
    self,
    model_size: str = "small",
    device: str = "auto",
    logger: logging.Logger | None = None,
  ) -> None:
    """Initialize LocalWhisperClient.

    Args:
      model_size: Whisper model size (tiny, base, small, medium, large-v3)
      device: Device to use (auto, cuda, cpu)
      logger: Logger instance for output logging
    """
    self.model_size = model_size
    self.device = device
    self.logger = logger or get_logger(__name__)
    self._model: "WhisperModel | None" = None

  def _detect_device(self) -> tuple[str, str]:
    """Detect optimal device and compute type.

    Returns:
      Tuple of (device, compute_type)
    """
    if self.device == "cpu":
      return "cpu", "int8"

    if self.device == "cuda":
      return "cuda", "float16"

    # Auto-detect: try CUDA first
    try:
      import torch

      if torch.cuda.is_available():
        self.logger.debug(f"CUDA available: {torch.cuda.get_device_name(0)}")
        return "cuda", "float16"
    except ImportError:
      pass

    self.logger.debug("CUDA not available, using CPU")
    return "cpu", "int8"

  def _setup_cuda_paths(self) -> None:
    """Set up CUDA library paths for faster-whisper."""
    # Check for nvidia libs in venv site-packages
    site_packages = None
    for path in sys.path:
      if "site-packages" in path and Path(path).exists():
        site_packages = Path(path)
        break

    if not site_packages:
      return

    # Look for NVIDIA CUDA libraries
    nvidia_paths = []
    for lib_dir in ["nvidia/cublas/lib", "nvidia/cudnn/lib"]:
      lib_path = site_packages / lib_dir
      if lib_path.exists():
        nvidia_paths.append(str(lib_path))

    if nvidia_paths:
      current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
      new_paths = ":".join(nvidia_paths)
      if current_ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{new_paths}:{current_ld_path}"
      else:
        os.environ["LD_LIBRARY_PATH"] = new_paths
      self.logger.debug(f"Added CUDA paths: {new_paths}")

      # LD_LIBRARY_PATH set after process start is not reliably honored by the
      # glibc loader for the current process, so dlopen the discovered cublas/
      # cudnn libraries explicitly (RTLD_GLOBAL) to expose their symbols to
      # ctranslate2. Best-effort: never crash if a library is missing.
      import ctypes

      for lib_dir in nvidia_paths:
        for so_file in sorted(Path(lib_dir).glob("*.so*")):
          try:
            ctypes.CDLL(str(so_file), mode=ctypes.RTLD_GLOBAL)
          except OSError as e:
            self.logger.debug(f"Could not preload {so_file}: {e}")

  def _get_model(self) -> "WhisperModel":
    """Lazy-load the Whisper model."""
    if self._model is not None:
      return self._model

    # Set up CUDA paths before importing faster_whisper
    self._setup_cuda_paths()

    from faster_whisper import WhisperModel

    device, compute_type = self._detect_device()
    self.logger.debug(f"Loading model: {self.model_size} on {device} ({compute_type})")

    try:
      self._model = WhisperModel(
        self.model_size,
        device=device,
        compute_type=compute_type,
      )
    except RuntimeError as e:
      # faster-whisper/ctranslate2 loads CUDA via cuDNN/cuBLAS, whose
      # availability torch.cuda.is_available() does not guarantee. When the
      # device was auto-detected, degrade gracefully to CPU; when the user
      # explicitly requested cuda, surface the error.
      if device == "cuda" and self.device == "auto":
        self.logger.warning(
          f"CUDA model load failed ({e}); falling back to CPU. "
          "For faster local transcription, ensure cuDNN/cuBLAS are installed."
        )
        self._model = WhisperModel(
          self.model_size,
          device="cpu",
          compute_type="int8",
        )
      else:
        raise
    return self._model

  def _convert_to_openai_format(self, segments: list, info: Any) -> dict:
    """Convert faster-whisper output to OpenAI verbose_json format.

    Args:
      segments: List of segment objects from faster-whisper
      info: TranscriptionInfo object from faster-whisper

    Returns:
      Dictionary matching OpenAI's verbose_json format
    """
    return {
      "text": " ".join(seg.text.strip() for seg in segments),
      "language": info.language,
      "duration": info.duration,
      "segments": [
        {
          "id": i,
          "start": seg.start,
          "end": seg.end,
          "text": seg.text.strip(),
          "words": [
            {"word": w.word, "start": w.start, "end": w.end}
            for w in (seg.words or [])
          ],
        }
        for i, seg in enumerate(segments)
      ],
    }

  def transcribe_audio(
    self,
    audio_file: str | BinaryIO,
    model: str = "whisper-1",
    prompt: str = "",
    language: str = "en",
    response_format: str = "text",
    temperature: float = 0.05,
  ) -> str | dict:
    """Transcribe audio using local faster-whisper model.

    OpenAI-compatible interface for transcription.

    Args:
      audio_file: Path to audio file or file-like object
      model: Ignored (uses self.model_size instead)
      prompt: Optional context prompt to guide transcription
      language: Language code (e.g., 'en', 'fr')
      response_format: Format of response ("text" or "verbose_json")
      temperature: Temperature for generation

    Returns:
      Transcription result in requested format:
        - "text": Plain text string
        - "verbose_json": Dictionary with segments, timing, language

    Raises:
      FileNotFoundError: If audio file path doesn't exist
      ValueError: If audio_file is a file object (not supported yet)
    """
    # Handle file path vs file object
    if isinstance(audio_file, str):
      audio_path = Path(audio_file)
      if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    else:
      raise ValueError("File objects not yet supported; provide a file path")

    model_instance = self._get_model()
    self.logger.debug(f"Transcribing: {audio_path}")

    # Build transcription parameters
    params: dict[str, Any] = {
      "beam_size": 5,
      "vad_filter": True,
      "language": language if language else None,
      "initial_prompt": prompt if prompt else None,
      "temperature": temperature,
    }

    # Request word timestamps for verbose_json format
    if response_format == "verbose_json":
      params["word_timestamps"] = True

    segments_gen, info = model_instance.transcribe(str(audio_path), **params)

    # Consume generator to list
    segments = list(segments_gen)

    if response_format == "verbose_json":
      return self._convert_to_openai_format(segments, info)

    # Default: return plain text
    return " ".join(seg.text.strip() for seg in segments)


#fin
