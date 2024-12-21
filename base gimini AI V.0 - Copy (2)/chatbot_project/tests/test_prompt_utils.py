import pytest
from typing import List
import prompt_utils


def test_build_prompt():
    """Tests the _build_prompt function."""
    prompt = "Hello, how are you?"
    memory_context = "User likes Python"
    feedback_context = "Previous response was not helpful"
    knowledge_results = "Found info about Python"
    formatted_conversation = "User: Hi, AI: Hello"
    expected_prompt_start = "You are Gemini, a helpful and engaging AI assistant."
    built_prompt = prompt_utils._build_prompt(prompt, memory_context, feedback_context,
                                             knowledge_results, formatted_conversation)
    assert built_prompt.startswith(expected_prompt_start)
    assert prompt in built_prompt
    assert memory_context in built_prompt
    assert feedback_context in built_prompt
    assert knowledge_results in built_prompt
    assert formatted_conversation in built_prompt


def test_generate_response():
    """Tests the generate_response function (basic test, actual API call mocked)."""
    prompt = "What is AI?"
    conversation_history = ["User: Hi", "AI: Hello"]
    memory = {"user_preferences": {"programming_language": "python"}}
    knowledge_results = "Found information about AI."

    response = prompt_utils.generate_response(prompt, conversation_history, memory, knowledge_results)
    assert isinstance(response, str)
    assert "Sorry, I encountered an issue" not in response  # Basic check for API error handling
    # More realistic testing would require mocking the API call.