import pytest

from llm_assistant.conversation import Conversation


@pytest.fixture
def conversation() -> Conversation:
    return Conversation()


def test_new_conversation_is_empty(conversation):
    assert conversation.messages == []


def test_add_user_message(conversation):
    conversation.add_user_message("Hello")

    assert conversation.messages == [
        {"role": "user", "content": "Hello"}
    ]

def test_add_assistant_message(conversation):
    conversation.add_assistant_message("World")

    assert conversation.messages == [
        {"role": "assistant", "content": "World"}
    ]

def test_remove_last_message_from_empty(conversation):
    conversation.remove_last_message()
    assert conversation.messages == []

def test_remove_last_message(conversation):
    conversation.add_user_message("Hello")
    conversation.remove_last_message()
    assert conversation.messages == []

def test_get_messages(conversation):
    conversation.add_user_message("Hello")
    copy: list[dict[str, str]] = conversation.get_messages()
    assert conversation.messages is not copy
    assert conversation.messages == copy

def test_clear(conversation):
    conversation.add_user_message("Hello")
    conversation.add_assistant_message("World")
    conversation.clear()
    assert conversation.messages == []
