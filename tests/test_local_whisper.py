#!/usr/bin/env python3
"""Tests for LocalWhisperClient adapter."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transcribe_pkg.utils.local_whisper import LocalWhisperClient


class MockSegment:
  """Mock segment matching faster-whisper Segment structure."""

  def __init__(self, id: int, start: float, end: float, text: str, words=None):
    self.id = id
    self.start = start
    self.end = end
    self.text = text
    self.words = words or []


class MockWord:
  """Mock word matching faster-whisper Word structure."""

  def __init__(self, word: str, start: float, end: float):
    self.word = word
    self.start = start
    self.end = end


class MockTranscriptionInfo:
  """Mock transcription info matching faster-whisper structure."""

  def __init__(self, language: str = "en", duration: float = 10.5):
    self.language = language
    self.duration = duration
    self.language_probability = 0.98


class TestLocalWhisperClientInit:
  """Tests for LocalWhisperClient initialization."""

  def test_defaults(self):
    """Test default initialization values."""
    client = LocalWhisperClient()
    assert client.model_size == "small"
    assert client.device == "auto"
    assert client._model is None

  def test_custom_params(self):
    """Test custom initialization parameters."""
    client = LocalWhisperClient(model_size="large-v3", device="cuda")
    assert client.model_size == "large-v3"
    assert client.device == "cuda"


class TestDeviceDetection:
  """Tests for device detection logic."""

  def test_cpu_explicit(self):
    """Test explicit CPU device selection."""
    client = LocalWhisperClient(device="cpu")
    device, compute_type = client._detect_device()
    assert device == "cpu"
    assert compute_type == "int8"

  def test_cuda_explicit(self):
    """Test explicit CUDA device selection."""
    client = LocalWhisperClient(device="cuda")
    device, compute_type = client._detect_device()
    assert device == "cuda"
    assert compute_type == "float16"

  @patch("transcribe_pkg.utils.local_whisper.LocalWhisperClient._detect_device")
  def test_auto_with_cuda_available(self, mock_detect):
    """Test auto-detection when CUDA is available."""
    mock_detect.return_value = ("cuda", "float16")
    client = LocalWhisperClient(device="auto")
    device, compute_type = client._detect_device()
    assert device == "cuda"
    assert compute_type == "float16"

  @patch("transcribe_pkg.utils.local_whisper.LocalWhisperClient._detect_device")
  def test_auto_cpu_fallback(self, mock_detect):
    """Test auto-detection falling back to CPU."""
    mock_detect.return_value = ("cpu", "int8")
    client = LocalWhisperClient(device="auto")
    device, compute_type = client._detect_device()
    assert device == "cpu"
    assert compute_type == "int8"


class TestTranscribeAudio:
  """Tests for transcribe_audio method."""

  @pytest.fixture
  def client(self):
    """Create a client with mocked model."""
    return LocalWhisperClient(model_size="small", device="cpu")

  @pytest.fixture
  def mock_segments(self):
    """Create mock segments for testing."""
    return [
      MockSegment(0, 0.0, 2.5, " Hello world. "),
      MockSegment(1, 2.5, 5.0, " This is a test. "),
    ]

  @pytest.fixture
  def mock_segments_with_words(self):
    """Create mock segments with word-level timestamps."""
    return [
      MockSegment(
        0, 0.0, 2.5, " Hello world. ",
        words=[
          MockWord("Hello", 0.0, 1.2),
          MockWord("world.", 1.2, 2.5),
        ],
      ),
      MockSegment(
        1, 2.5, 5.0, " This is a test. ",
        words=[
          MockWord("This", 2.5, 3.0),
          MockWord("is", 3.0, 3.3),
          MockWord("a", 3.3, 3.5),
          MockWord("test.", 3.5, 5.0),
        ],
      ),
    ]

  @pytest.fixture
  def mock_info(self):
    """Create mock transcription info."""
    return MockTranscriptionInfo(language="en", duration=5.0)

  def test_file_not_found(self, client):
    """Test error when audio file doesn't exist."""
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
      client.transcribe_audio("/nonexistent/file.mp3")

  def test_file_object_not_supported(self, client):
    """Test error when file object is provided."""
    mock_file = MagicMock()
    with pytest.raises(ValueError, match="File objects not yet supported"):
      client.transcribe_audio(mock_file)

  @patch("transcribe_pkg.utils.local_whisper.LocalWhisperClient._get_model")
  def test_transcribe_text_format(
    self, mock_get_model, client, mock_segments, mock_info, tmp_path,
  ):
    """Test transcription with text response format."""
    # Create temp audio file
    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio data")

    # Mock the model
    mock_model = MagicMock()
    mock_model.transcribe.return_value = (iter(mock_segments), mock_info)
    mock_get_model.return_value = mock_model

    result = client.transcribe_audio(str(audio_file), response_format="text")

    assert isinstance(result, str)
    assert result == "Hello world. This is a test."

  @patch("transcribe_pkg.utils.local_whisper.LocalWhisperClient._get_model")
  def test_transcribe_verbose_json_format(
    self,
    mock_get_model,
    client,
    mock_segments_with_words,
    mock_info,
    tmp_path,
  ):
    """Test transcription with verbose_json response format."""
    # Create temp audio file
    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio data")

    # Mock the model
    mock_model = MagicMock()
    mock_model.transcribe.return_value = (iter(mock_segments_with_words), mock_info)
    mock_get_model.return_value = mock_model

    result = client.transcribe_audio(str(audio_file), response_format="verbose_json")

    assert isinstance(result, dict)
    assert "text" in result
    assert "language" in result
    assert "duration" in result
    assert "segments" in result

    assert result["text"] == "Hello world. This is a test."
    assert result["language"] == "en"
    assert result["duration"] == 5.0
    assert len(result["segments"]) == 2

    # Check first segment structure
    seg0 = result["segments"][0]
    assert seg0["id"] == 0
    assert seg0["start"] == 0.0
    assert seg0["end"] == 2.5
    assert seg0["text"] == "Hello world."
    assert len(seg0["words"]) == 2

    # Check word structure
    word0 = seg0["words"][0]
    assert word0["word"] == "Hello"
    assert word0["start"] == 0.0
    assert word0["end"] == 1.2

  @patch("transcribe_pkg.utils.local_whisper.LocalWhisperClient._get_model")
  def test_transcribe_with_language(
    self, mock_get_model, client, mock_segments, mock_info, tmp_path,
  ):
    """Test transcription with language parameter."""
    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio data")

    mock_model = MagicMock()
    mock_model.transcribe.return_value = (iter(mock_segments), mock_info)
    mock_get_model.return_value = mock_model

    client.transcribe_audio(str(audio_file), language="fr")

    # Verify language was passed to model
    call_args = mock_model.transcribe.call_args
    assert call_args[1]["language"] == "fr"

  @patch("transcribe_pkg.utils.local_whisper.LocalWhisperClient._get_model")
  def test_transcribe_with_prompt(
    self, mock_get_model, client, mock_segments, mock_info, tmp_path,
  ):
    """Test transcription with initial prompt."""
    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio data")

    mock_model = MagicMock()
    mock_model.transcribe.return_value = (iter(mock_segments), mock_info)
    mock_get_model.return_value = mock_model

    client.transcribe_audio(str(audio_file), prompt="Technical discussion about Python")

    # Verify prompt was passed to model
    call_args = mock_model.transcribe.call_args
    assert call_args[1]["initial_prompt"] == "Technical discussion about Python"


class TestConvertToOpenAIFormat:
  """Tests for format conversion."""

  def test_empty_segments(self):
    """Test conversion with no segments."""
    client = LocalWhisperClient()
    info = MockTranscriptionInfo()

    result = client._convert_to_openai_format([], info)

    assert result["text"] == ""
    assert result["segments"] == []
    assert result["language"] == "en"

  def test_segments_without_words(self):
    """Test conversion when segments have no word timestamps."""
    client = LocalWhisperClient()
    info = MockTranscriptionInfo()
    segments = [MockSegment(0, 0.0, 1.0, " Test ")]

    result = client._convert_to_openai_format(segments, info)

    assert result["segments"][0]["words"] == []


class TestCudaPathSetup:
  """Tests for CUDA path setup."""

  def test_setup_cuda_paths_no_nvidia_libs(self):
    """Test CUDA path setup when no nvidia libs exist."""
    client = LocalWhisperClient()
    # Should not raise even if no nvidia libs found
    client._setup_cuda_paths()

  def test_setup_cuda_paths_loads_libs_via_ctypes(self, tmp_path, monkeypatch):
    """Discovered cublas/cudnn .so files are dlopen'd with RTLD_GLOBAL.

    LD_LIBRARY_PATH mutation after process start is not honored by the
    loader; the libraries must be loaded explicitly via ctypes so the
    symbols are available to ctranslate2.
    """
    # Build a fake site-packages tree with nvidia cublas/cudnn .so files.
    site_packages = tmp_path / "site-packages"
    cublas_dir = site_packages / "nvidia" / "cublas" / "lib"
    cudnn_dir = site_packages / "nvidia" / "cudnn" / "lib"
    cublas_dir.mkdir(parents=True)
    cudnn_dir.mkdir(parents=True)
    cublas_so = cublas_dir / "libcublas.so.12"
    cudnn_so = cudnn_dir / "libcudnn.so.9"
    cublas_so.write_bytes(b"")
    cudnn_so.write_bytes(b"")

    monkeypatch.setattr(
      "transcribe_pkg.utils.local_whisper.sys.path",
      [str(site_packages)],
    )

    loaded = []
    fake_ctypes = MagicMock()
    fake_ctypes.RTLD_GLOBAL = 256

    def fake_cdll(path, mode=0):
      loaded.append((path, mode))
      return MagicMock()

    fake_ctypes.CDLL.side_effect = fake_cdll

    client = LocalWhisperClient()
    with patch.dict("sys.modules", {"ctypes": fake_ctypes}):
      client._setup_cuda_paths()

    loaded_paths = [p for p, _ in loaded]
    assert str(cublas_so) in loaded_paths
    assert str(cudnn_so) in loaded_paths
    # Must be loaded globally so dependent libs resolve the symbols.
    for _, mode in loaded:
      assert mode == fake_ctypes.RTLD_GLOBAL

  def test_setup_cuda_paths_ctypes_failure_does_not_crash(self, tmp_path, monkeypatch):
    """A failing/missing .so load is swallowed (best-effort)."""
    site_packages = tmp_path / "site-packages"
    cublas_dir = site_packages / "nvidia" / "cublas" / "lib"
    cublas_dir.mkdir(parents=True)
    (cublas_dir / "libcublas.so.12").write_bytes(b"")

    monkeypatch.setattr(
      "transcribe_pkg.utils.local_whisper.sys.path",
      [str(site_packages)],
    )

    fake_ctypes = MagicMock()
    fake_ctypes.RTLD_GLOBAL = 256
    fake_ctypes.CDLL.side_effect = OSError("cannot open shared object file")

    client = LocalWhisperClient()
    with patch.dict("sys.modules", {"ctypes": fake_ctypes}):
      # Should not raise despite the OSError from CDLL.
      client._setup_cuda_paths()


class TestGpuFallback:
  """Tests for graceful CPU fallback when CUDA model load fails."""

  @patch("transcribe_pkg.utils.local_whisper.LocalWhisperClient._setup_cuda_paths")
  @patch("transcribe_pkg.utils.local_whisper.LocalWhisperClient._detect_device")
  def test_auto_cuda_load_failure_falls_back_to_cpu(
    self, mock_detect, mock_setup, monkeypatch,
  ):
    """When device=auto resolves to cuda but model load fails, fall back to CPU."""
    mock_detect.return_value = ("cuda", "float16")

    cpu_model = object()
    calls = []

    def fake_whisper_model(model_size, device, compute_type):
      calls.append((device, compute_type))
      if device == "cuda":
        raise RuntimeError("Unable to load libcudnn_ops.so.9")
      return cpu_model

    fake_module = MagicMock()
    fake_module.WhisperModel.side_effect = fake_whisper_model

    client = LocalWhisperClient(model_size="small", device="auto")
    with patch.dict("sys.modules", {"faster_whisper": fake_module}):
      result = client._get_model()

    assert result is cpu_model
    # cuda attempted first, then cpu/int8 fallback.
    assert calls[0] == ("cuda", "float16")
    assert ("cpu", "int8") in calls

  @patch("transcribe_pkg.utils.local_whisper.LocalWhisperClient._setup_cuda_paths")
  @patch("transcribe_pkg.utils.local_whisper.LocalWhisperClient._detect_device")
  def test_explicit_cuda_load_failure_reraises(
    self, mock_detect, mock_setup,
  ):
    """When user explicitly requests device=cuda, a load failure must propagate."""
    mock_detect.return_value = ("cuda", "float16")

    def fake_whisper_model(model_size, device, compute_type):
      raise RuntimeError("Unable to load libcudnn_ops.so.9")

    fake_module = MagicMock()
    fake_module.WhisperModel.side_effect = fake_whisper_model

    client = LocalWhisperClient(model_size="small", device="cuda")
    with patch.dict("sys.modules", {"faster_whisper": fake_module}):
      with pytest.raises(RuntimeError, match="libcudnn"):
        client._get_model()


@pytest.mark.integration
class TestIntegration:
  """Integration tests using real audio file (skip in CI)."""

  @pytest.fixture
  def test_audio_file(self):
    """Get path to test audio file."""
    test_file = Path(__file__).parent / "test.mp3"
    if not test_file.exists():
      pytest.skip("Test audio file not found")
    return str(test_file)

  @pytest.mark.skip(reason="Requires GPU/model download - run manually")
  def test_real_transcription_text(self, test_audio_file):
    """Test real transcription with text format."""
    client = LocalWhisperClient(model_size="tiny", device="auto")
    result = client.transcribe_audio(test_audio_file, response_format="text")

    assert isinstance(result, str)
    assert len(result) > 0

  @pytest.mark.skip(reason="Requires GPU/model download - run manually")
  def test_real_transcription_verbose_json(self, test_audio_file):
    """Test real transcription with verbose_json format."""
    client = LocalWhisperClient(model_size="tiny", device="auto")
    result = client.transcribe_audio(test_audio_file, response_format="verbose_json")

    assert isinstance(result, dict)
    assert "text" in result
    assert "segments" in result
    assert len(result["segments"]) > 0


#fin
