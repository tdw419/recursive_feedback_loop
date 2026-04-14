"""Tests for the loop runner."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from recursive_feedback_loop.config import LoopConfig
from recursive_feedback_loop.loop_runner import LoopRunner, LoopState
from recursive_feedback_loop.session_reader import Conversation


def test_loop_state_logging():
    with tempfile.TemporaryDirectory() as td:
        state = LoopState(Path(td))
        state.log("test_event", {"key": "value"})
        state.log("another_event")

        log_entries = []
        with open(state.log_file) as f:
            for line in f:
                log_entries.append(json.loads(line))

        assert len(log_entries) == 2
        assert log_entries[0]["event"] == "test_event"
        assert log_entries[0]["data"]["key"] == "value"
        assert log_entries[1]["event"] == "another_event"


def test_loop_state_snapshot():
    with tempfile.TemporaryDirectory() as td:
        state = LoopState(Path(td))
        state.conversation.add("user", "Hi", iteration=0)
        state.conversation.add("assistant", "Hello", iteration=0)
        state.iteration = 1

        snap_path = state.save_snapshot()
        assert snap_path.exists()

        data = json.loads(snap_path.read_text())
        assert data["iteration"] == 1
        assert len(data["turns"]) == 2


def test_loop_state_export_markdown():
    with tempfile.TemporaryDirectory() as td:
        state = LoopState(Path(td))
        state.conversation.add("user", "What is consciousness?", iteration=0)
        state.conversation.add("assistant", "A complex phenomenon", iteration=0)

        md_path = state.export_markdown()
        assert md_path.exists()
        content = md_path.read_text()
        assert "USER" in content
        assert "consciousness" in content


def test_loop_state_export_jsonl():
    with tempfile.TemporaryDirectory() as td:
        state = LoopState(Path(td))
        state.conversation.add("user", "Hi", iteration=0)

        jsonl_path = state.export_jsonl()
        assert jsonl_path.exists()
        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["role"] == "user"


def test_loop_runner_with_mock_hermes():
    """Test the full loop with mocked Hermes calls."""
    with tempfile.TemporaryDirectory() as td:
        config = LoopConfig(
            seed_prompt="Start thinking about recursion",
            max_iterations=3,
            compaction_strategy="sliding_window",
            output_dir=td,
            export_format="both",
        )

        # Track call count to generate different responses
        call_count = {"n": 0}

        def mock_run(args, **kwargs):
            call_count["n"] += 1
            mock = MagicMock()
            mock.returncode = 0
            mock.stdout = f"Response iteration {call_count['n']}: thinking about recursion at depth {call_count['n']}"
            mock.stderr = ""
            return mock

        with patch("subprocess.run", side_effect=mock_run):
            runner = LoopRunner(config)
            state = runner.run()

        assert len(state.conversation) == 4  # 1 seed user + 3 assistant responses
        assert state.conversation.turns[0].role == "user"
        assert state.conversation.turns[1].role == "assistant"
        assert state.conversation.turns[1].content == "Response iteration 1: thinking about recursion at depth 1"

        # Check output files exist
        assert (state.output_dir / "conversation.md").exists()
        assert (state.output_dir / "conversation.jsonl").exists()
        assert (state.output_dir / "loop_log.jsonl").exists()

        # Check snapshots
        snaps = list((state.output_dir / "snapshots").glob("iter_*.json"))
        assert len(snaps) > 0


def test_loop_runner_elapsed():
    """Test that elapsed time is tracked."""
    with tempfile.TemporaryDirectory() as td:
        state = LoopState(Path(td))
        import time
        time.sleep(0.1)
        assert state.elapsed() >= 0.1
