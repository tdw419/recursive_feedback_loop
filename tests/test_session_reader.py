"""Tests for session reader."""

import json
import tempfile
from pathlib import Path

from recursive_feedback_loop.session_reader import SessionReader, Turn, Conversation


def test_turn_creation():
    t = Turn(role="user", content="Hello", timestamp="2026-01-01T00:00:00", iteration=0)
    assert t.role == "user"
    assert t.content == "Hello"
    assert t.token_estimate() > 0


def test_turn_repr():
    t = Turn(role="assistant", content="A" * 200, iteration=1)
    r = repr(t)
    assert "assistant" in r
    assert "iter=1" in r


def test_conversation():
    c = Conversation()
    c.add("user", "Hi", iteration=0)
    c.add("assistant", "Hello!", iteration=0)
    assert len(c) == 2
    assert c.total_tokens_estimate() > 0


def test_conversation_last_n():
    c = Conversation()
    for i in range(10):
        c.add("user", f"Turn {i}", iteration=i)
    last = c.last_n_turns(3)
    assert len(last) == 3
    assert last[0].content == "Turn 7"


def test_conversation_by_iteration():
    c = Conversation()
    c.add("user", "A", iteration=0)
    c.add("assistant", "B", iteration=0)
    c.add("user", "C", iteration=1)
    assert len(c.turns_by_iteration(0)) == 2
    assert len(c.turns_by_iteration(1)) == 1


def test_session_reader():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"role": "user", "content": "Hello", "timestamp": "2026-01-01T00:00:00"}) + "\n")
        f.write(json.dumps({"role": "assistant", "content": "Hi there!", "timestamp": "2026-01-01T00:00:01"}) + "\n")
        f.write(json.dumps({"role": "tool", "content": "tool output", "timestamp": "2026-01-01T00:00:02"}) + "\n")
        f.write(json.dumps({"role": "user", "content": "How are you?", "timestamp": "2026-01-01T00:00:03"}) + "\n")
        path = f.name

    reader = SessionReader(path)
    turns = reader.read_new_turns()
    # Should get 3 turns (user, assistant, user) — tool filtered out
    assert len(turns) == 3
    assert turns[0].role == "user"
    assert turns[1].role == "assistant"
    assert turns[2].role == "user"

    # Second read should return nothing new
    turns2 = reader.read_new_turns()
    assert len(turns2) == 0


def test_session_reader_incremental():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"role": "user", "content": "First", "timestamp": "2026-01-01T00:00:00"}) + "\n")
        path = f.name

    reader = SessionReader(path)
    turns1 = reader.read_new_turns()
    assert len(turns1) == 1

    # Append more
    with open(path, "a") as f:
        f.write(json.dumps({"role": "assistant", "content": "Second", "timestamp": "2026-01-01T00:00:01"}) + "\n")

    turns2 = reader.read_new_turns()
    assert len(turns2) == 1
    assert turns2[0].role == "assistant"


def test_read_all():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for i in range(5):
            f.write(json.dumps({"role": "user", "content": f"Turn {i}", "timestamp": "2026-01-01T00:00:00"}) + "\n")
        path = f.name

    reader = SessionReader(path)
    conv = reader.read_all()
    assert len(conv) == 5


def test_extract_assistant_response():
    from recursive_feedback_loop.session_reader import extract_assistant_response
    turns = [
        Turn(role="user", content="Q", iteration=0),
        Turn(role="assistant", content="A1", iteration=0),
        Turn(role="user", content="Q2", iteration=1),
        Turn(role="assistant", content="A2", iteration=1),
    ]
    assert extract_assistant_response(turns) == "A2"
    assert extract_assistant_response(turns[:1]) is None


def test_to_dicts():
    c = Conversation()
    c.add("user", "Hi", iteration=0)
    c.add("assistant", "Hello", iteration=0)
    dicts = c.to_dicts()
    assert len(dicts) == 2
    assert dicts[0]["role"] == "user"
    assert dicts[1]["iteration"] == 0
