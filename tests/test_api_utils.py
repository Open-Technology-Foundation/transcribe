#!/usr/bin/env python3
"""
Tests for the API utilities module.
"""
import os
import unittest
from unittest.mock import patch, MagicMock

from transcribe_pkg.utils.api_utils import (
    OpenAIClient,
    get_openai_client,
    call_llm,
    transcribe_audio,
    APIError,
    EmptyResponseError
)

class TestAPIUtils(unittest.TestCase):
    """Tests for API utility functions."""

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-api-key'})
    @patch('openai.OpenAI')
    def test_get_openai_client(self, mock_openai):
        """Test getting OpenAI client with API key (SDK client built lazily on use)."""
        # Reset globals for test isolation
        import transcribe_pkg.utils.api_utils as api_utils_module
        api_utils_module._global_client = None
        api_utils_module.openai_client = None

        # Setup mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Call function
        client = get_openai_client()

        # Check results - client is an OpenAIClient wrapper, not the raw openai client
        self.assertIsNotNone(client)
        # The underlying SDK client is constructed lazily on first access
        self.assertIs(client.client, mock_client)
        mock_openai.assert_called_once_with(api_key='test-api-key')
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': ''})  # Empty string instead of completely missing
    def test_get_openai_client_no_key(self):
        """Without a key, getting the client is lazy; only USING it raises ValueError."""
        # Clear global client to ensure test isolation
        import transcribe_pkg.utils.api_utils as api_utils_module
        api_utils_module._global_client = None
        api_utils_module.openai_client = None

        # Getting the client must NOT raise (construction is lazy)
        try:
            client = get_openai_client()
        except ValueError:
            self.fail("get_openai_client() must not require a key until the client is used")

        # Using the underlying SDK client without a key must raise
        with self.assertRaises(ValueError) as context:
            _ = client.client

        # Check error message
        self.assertIn("API key is required", str(context.exception))

    @patch.dict(os.environ, {'OPENAI_API_KEY': ''})
    def test_openai_client_construction_is_lazy_without_key(self):
        """OpenAIClient can be constructed without a key (e.g. as the default
        dependency of a non-OpenAI processor); the key is only needed on use."""
        try:
            OpenAIClient()
        except ValueError:
            self.fail("OpenAIClient construction must be lazy and not require a key")
    
    @patch('transcribe_pkg.utils.api_utils.openai_client')
    def test_call_llm_success(self, mock_client):
        """Test successful LLM API call."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response
        
        # Call function
        result = call_llm("User input", "System prompt", model="test-model")
        
        # Check results
        self.assertEqual(result, "Test response")
        mock_client.chat.completions.create.assert_called_once()
        args, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs['model'], "test-model")
        self.assertEqual(len(kwargs['messages']), 2)
    
    @patch('transcribe_pkg.utils.api_utils.openai_client')
    def test_call_llm_empty_response(self, mock_client):
        """Test LLM API call with empty response."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""
        mock_client.chat.completions.create.return_value = mock_response
        
        # Call function
        result = call_llm("User input", "System prompt")
        
        # Check results
        self.assertEqual(result, "")
    
    @patch('transcribe_pkg.utils.api_utils.openai_client')
    @patch('transcribe_pkg.utils.api_utils.logging')
    def test_call_llm_no_choices(self, mock_logging, mock_client):
        """Test LLM API call with no choices."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = []
        mock_client.chat.completions.create.return_value = mock_response
        
        try:
            # Call the main function with new parameter order
            call_llm("User input", "System prompt")
            self.fail("EmptyResponseError not raised")
        except EmptyResponseError:
            # Success - error was raised as expected
            pass
    
    @patch('transcribe_pkg.utils.api_utils.openai_client')
    @patch('transcribe_pkg.utils.api_utils.logging')
    def test_call_llm_api_error(self, mock_logging, mock_client):
        """Test LLM API call with API error."""
        # Setup mock to raise error
        mock_client.chat.completions.create.side_effect = Exception("API error")
        
        try:
            # Call the main function with new parameter order
            call_llm("User input", "System prompt")
            self.fail("APIError not raised")
        except APIError:
            # Success - error was raised as expected
            pass
    
    @patch('os.path.exists')
    @patch('transcribe_pkg.utils.api_utils.openai_client')
    def test_transcribe_audio_success(self, mock_client, mock_exists):
        """Test successful audio transcription."""
        # Setup mocks
        mock_exists.return_value = True
        mock_transcription = MagicMock()
        mock_transcription.text = "Test transcription"
        mock_client.audio.transcriptions.create.return_value = mock_transcription
        
        # Mock file open
        mock_file = MagicMock()
        mock_file.__enter__.return_value = "audio_data"
        
        with patch('builtins.open', return_value=mock_file):
            # Call function
            result = transcribe_audio("test.mp3", prompt="Test prompt")
        
        # Check results
        self.assertEqual(result, "Test transcription")
        mock_client.audio.transcriptions.create.assert_called_once()
    
    @patch('os.path.exists')
    def test_transcribe_audio_file_not_found(self, mock_exists):
        """Test audio transcription with file not found."""
        # Setup mock
        mock_exists.return_value = False
        
        try:
            # Call the implementation function directly to avoid retry
            from transcribe_pkg.utils.api_utils import _transcribe_audio_impl
            _transcribe_audio_impl("nonexistent.mp3")
            self.fail("FileNotFoundError not raised")
        except FileNotFoundError:
            # Success - error was raised as expected
            pass
    
    @patch('os.path.exists')
    @patch('os.path.getsize')
    def test_transcribe_audio_file_too_large(self, mock_getsize, mock_exists):
        """Test audio transcription with file size - just verify it works."""
        # Setup mocks
        mock_exists.return_value = True
        mock_getsize.return_value = 30 * 1024 * 1024  # 30MB

        # Need to mock the actual transcription to avoid real API call
        mock_transcription = MagicMock()
        mock_transcription.text = "Test transcription"

        with patch('transcribe_pkg.utils.api_utils.openai_client') as mock_client:
            mock_client.audio.transcriptions.create.return_value = mock_transcription

            # Mock file open
            with patch('builtins.open', MagicMock()):
                # Call function - should work despite large size
                result = transcribe_audio("large.mp3")

        # Verify transcription still works
        self.assertEqual(result, "Test transcription")


class TestDefaultReasoningEffort(unittest.TestCase):
    """`_default_reasoning_effort` must return an SDK-accepted reasoning_effort.

    The installed OpenAI SDK's ReasoningEffort literal accepts only the values in
    VALID_REASONING_EFFORTS; "none" / "xhigh" are rejected by the API with a 400.
    The default must stay inside that set, or be None to omit the parameter.
    """

    # Valid for openai>=2.1 (ReasoningEffort = Literal["minimal","low","medium","high"]).
    VALID_REASONING_EFFORTS = {"minimal", "low", "medium", "high"}

    def test_reasoning_model_default_is_sdk_valid(self):
        """A reasoning model gets a default effort the SDK actually accepts."""
        from transcribe_pkg.utils.api_utils import _default_reasoning_effort
        effort = _default_reasoning_effort("gpt-5")
        self.assertIn(
            effort, self.VALID_REASONING_EFFORTS,
            f"reasoning_effort {effort!r} is not a valid OpenAI ReasoningEffort value"
        )

    def test_non_reasoning_model_omits_effort(self):
        """A non-reasoning model returns None so the parameter is omitted."""
        from transcribe_pkg.utils.api_utils import _default_reasoning_effort
        self.assertIsNone(_default_reasoning_effort("gpt-4o"))


class TestChatCompletionNoneContent(unittest.TestCase):
    """`OpenAIClient.chat_completion` must not crash when a reasoning model
    returns `None` content (ChatCompletionMessage.content is Optional[str])."""

    def _make_client_with_content(self, content):
        client = OpenAIClient(api_key="test-key")
        mock_sdk = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = content
        mock_sdk.chat.completions.create.return_value = mock_response
        client._client = mock_sdk
        return client

    def test_none_content_returns_empty_string(self):
        """None content (common for reasoning models) returns '' not AttributeError."""
        client = self._make_client_with_content(None)
        result = client.chat_completion(
            system_prompt="sys", user_prompt="hi", model="gpt-5"
        )
        self.assertEqual(result, "")

    def test_happy_path_still_strips_content(self):
        """Non-None content is still returned, stripped."""
        client = self._make_client_with_content("  hello  ")
        result = client.chat_completion(
            system_prompt="sys", user_prompt="hi", model="gpt-4o"
        )
        self.assertEqual(result, "hello")


class TestTranscribeAudioWrapperClient(unittest.TestCase):
    """Free `transcribe_audio()` must work when the global `openai_client` is an
    OpenAIClient WRAPPER (exposes the SDK as `.client`, not `.audio`)."""

    def test_wrapper_client_routes_to_transcribe_method(self):
        """A wrapper global (no `.audio`) must route through client.transcribe_audio()."""
        import transcribe_pkg.utils.api_utils as api_utils_module

        # Build a real wrapper whose SDK is mocked; it has `.client`, not `.audio`.
        # With response_format="text" the SDK returns a plain string, and the
        # wrapper.transcribe_audio() passes that through unchanged.
        wrapper = OpenAIClient(api_key="test-key")
        mock_sdk = MagicMock()
        mock_sdk.audio.transcriptions.create.return_value = "wrapper transcription"
        wrapper._client = mock_sdk

        self.assertFalse(hasattr(wrapper, "audio"))

        # The else-branch routes via get_openai_client(), which returns the
        # module-global _global_client; seed both globals with the wrapper.
        saved_client = api_utils_module.openai_client
        saved_global = api_utils_module._global_client
        api_utils_module.openai_client = wrapper
        api_utils_module._global_client = wrapper
        try:
            with patch("os.path.exists", return_value=True), \
                 patch("builtins.open", MagicMock()):
                result = transcribe_audio("test.mp3", prompt="p")
        finally:
            api_utils_module.openai_client = saved_client
            api_utils_module._global_client = saved_global

        self.assertEqual(result, "wrapper transcription")
        mock_sdk.audio.transcriptions.create.assert_called_once()


class TestTranscribeRetryPredicate(unittest.TestCase):
    """Transient timeouts (openai.APITimeoutError) must be retried, not dropped."""

    def test_apitimeouterror_is_retryable(self):
        """transcribe_audio's retry predicate must include openai.APITimeoutError."""
        import openai
        exc_types = OpenAIClient.transcribe_audio.retry.retry.exception_types
        self.assertIn(openai.APITimeoutError, exc_types)


class TestCallLLMImplNoneContent(unittest.TestCase):
    """`_call_llm_impl` must raise EmptyResponseError (not AttributeError) when
    the API returns None content."""

    def test_none_content_raises_empty_response_error(self):
        from transcribe_pkg.utils.api_utils import _call_llm_impl
        import transcribe_pkg.utils.api_utils as api_utils_module

        # Reset/seed the global wrapper with a mocked SDK returning None content.
        wrapper = OpenAIClient(api_key="test-key")
        mock_sdk = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_sdk.chat.completions.create.return_value = mock_response
        wrapper._client = mock_sdk

        saved_global = api_utils_module._global_client
        saved_client = api_utils_module.openai_client
        api_utils_module._global_client = wrapper
        api_utils_module.openai_client = wrapper
        try:
            with self.assertRaises(EmptyResponseError):
                _call_llm_impl("sys", "user", model="gpt-4o")
        finally:
            api_utils_module._global_client = saved_global
            api_utils_module.openai_client = saved_client


if __name__ == '__main__':
    unittest.main()

#fin