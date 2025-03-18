"""
Tests for osmosis-wrap functionality

This file tests the monkey patching functionality of osmosis-wrap for various LLM APIs.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

# Import osmosis_wrap and set up test environment
import osmosis_wrap

# Mock API key function to avoid environment variable requirements in tests
def mock_get_api_key(service_name):
    return f"mock-{service_name}-api-key"

# Initialize osmosis_wrap with a test API key
@pytest.fixture(scope="module")
def setup_osmosis():
    with patch("osmosis_wrap.utils.send_to_hoover") as mock_send_to_hoover:
        osmosis_wrap.init("test-hoover-api-key")
        yield mock_send_to_hoover

# Test Anthropic client wrapping
@pytest.mark.skipif(
    pytest.importorskip("anthropic", reason="anthropic package not installed") is None,
    reason="Anthropic package not installed"
)
def test_anthropic_wrapping(setup_osmosis):
    mock_send_to_hoover = setup_osmosis
    
    with patch("anthropic.Anthropic") as MockAnthropic:
        # Set up mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Mocked Anthropic Response")]
        MockAnthropic.return_value.messages.create.return_value = mock_response
        
        # Create client and make API call
        from anthropic import Anthropic
        client = Anthropic(api_key=mock_get_api_key("anthropic"))
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=150,
            messages=[
                {"role": "user", "content": "Test prompt"}
            ]
        )
        
        # Verify response is properly returned
        assert response.content[0].text == "Mocked Anthropic Response"
        
        # Verify hoover was called
        mock_send_to_hoover.assert_called_once()
        # Check that the query argument was passed
        assert "query" in mock_send_to_hoover.call_args[1]
        assert mock_send_to_hoover.call_args[1]["query"]["model"] == "claude-3-haiku-20240307"

# Test OpenAI v1 client wrapping
@pytest.mark.skipif(
    pytest.importorskip("openai", reason="openai package not installed") is None,
    reason="OpenAI package not installed"
)
def test_openai_v1_wrapping(setup_osmosis):
    mock_send_to_hoover = setup_osmosis
    mock_send_to_hoover.reset_mock()  # Reset call count
    
    # Mock the OpenAI package version to force v1 detection
    with patch("openai.version", __version__="1.0.0"), \
         patch("openai.OpenAI") as MockOpenAI:
        # Set up mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Mocked OpenAI Response"))]
        mock_response.model_dump = MagicMock(return_value={"choices": [{"message": {"content": "Mocked OpenAI Response"}}]})
        MockOpenAI.return_value.chat.completions.create.return_value = mock_response
        
        # Create client and make API call
        from openai import OpenAI
        client = OpenAI(api_key=mock_get_api_key("openai"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=150,
            messages=[
                {"role": "user", "content": "Test prompt for OpenAI v1"}
            ]
        )
        
        # Verify response is properly returned
        assert response.choices[0].message.content == "Mocked OpenAI Response"
        
        # Verify hoover was called
        mock_send_to_hoover.assert_called_once()
        # Check that the query argument was passed
        assert "query" in mock_send_to_hoover.call_args[1]
        assert mock_send_to_hoover.call_args[1]["query"]["model"] == "gpt-4o-mini"

# Test OpenAI v2 client wrapping
@pytest.mark.skipif(
    pytest.importorskip("openai", reason="openai package not installed") is None,
    reason="OpenAI package not installed"
)
def test_openai_v2_wrapping(setup_osmosis):
    mock_send_to_hoover = setup_osmosis
    mock_send_to_hoover.reset_mock()  # Reset call count
    
    # Mock the OpenAI package version to force v2 detection
    with patch("openai.version", __version__="2.0.0"), \
         patch("openai.OpenAI") as MockOpenAI:
        # Set up mock responses
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Mocked OpenAI v2 Response"))]
        mock_response.model_dump = MagicMock(return_value={"choices": [{"message": {"content": "Mocked OpenAI v2 Response"}}]})
        MockOpenAI.return_value.chat.completions.create.return_value = mock_response
        
        # Create mock streaming response
        mock_stream_chunks = [
            MagicMock(choices=[MagicMock(delta=MagicMock(content="Part 1 "))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="Part 2 "))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="Part 3"))])
        ]
        MockOpenAI.return_value.chat.completions.create.side_effect = [mock_response, mock_stream_chunks]
        
        # Create client and make standard API call
        from openai import OpenAI
        client = OpenAI(api_key=mock_get_api_key("openai"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=150,
            messages=[
                {"role": "user", "content": "Test prompt for OpenAI v2"}
            ]
        )
        
        # Verify response is properly returned
        assert response.choices[0].message.content == "Mocked OpenAI v2 Response"
        
        # Verify hoover was called
        assert mock_send_to_hoover.call_count == 1
        # Check that the query argument was passed
        assert "query" in mock_send_to_hoover.call_args[1]
        assert mock_send_to_hoover.call_args[1]["query"]["model"] == "gpt-4o-mini"
        
        # Reset mock for streaming test
        mock_send_to_hoover.reset_mock()
        
        # Test streaming API call
        MockOpenAI.return_value.chat.completions.create.return_value = mock_stream_chunks
        
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=50,
            messages=[{"role": "user", "content": "Test streaming"}],
            stream=True
        )
        
        # Simulate iteration over stream
        streamed_content = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                streamed_content += chunk.choices[0].delta.content
        
        # Verify streamed content
        assert streamed_content == "Part 1 Part 2 Part 3"
        
        # Verify hoover was called for the stream
        assert mock_send_to_hoover.call_count == 1
        # Check that the streaming parameter is captured
        assert "query" in mock_send_to_hoover.call_args[1]
        assert mock_send_to_hoover.call_args[1]["query"]["stream"] is True

# Test disabling hoover
def test_disable_hoover(setup_osmosis):
    mock_send_to_hoover = setup_osmosis
    mock_send_to_hoover.reset_mock()  # Reset call count
    
    # Test disabling and enabling
    osmosis_wrap.disable_hoover()
    assert not osmosis_wrap.utils.enabled
    
    # Make a mocked call with Anthropic
    with patch("anthropic.Anthropic") as MockAnthropic:
        # Set up mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Response while disabled")]
        MockAnthropic.return_value.messages.create.return_value = mock_response
        
        # Create client and make API call
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=mock_get_api_key("anthropic"))
            client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=150,
                messages=[{"role": "user", "content": "This request won't be logged."}]
            )
        except ImportError:
            # Fall back to OpenAI if Anthropic is not available
            with patch("openai.OpenAI") as MockOpenAI:
                mock_response = MagicMock()
                mock_response.choices = [MagicMock(message=MagicMock(content="Response while disabled"))]
                MockOpenAI.return_value.chat.completions.create.return_value = mock_response
                
                from openai import OpenAI
                client = OpenAI(api_key=mock_get_api_key("openai"))
                client.chat.completions.create(
                    model="gpt-4o-mini",
                    max_tokens=150,
                    messages=[{"role": "user", "content": "This request won't be logged."}]
                )
    
    # Verify hoover was not called
    assert not mock_send_to_hoover.called
    
    # Re-enable and test
    osmosis_wrap.enable_hoover()
    assert osmosis_wrap.utils.enabled

if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 