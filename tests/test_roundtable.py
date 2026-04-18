"""Tests for roundtable.py -- RoundTableConfig, RoundTableRunner."""

import json
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from recursive_feedback_loop.agents import AgentConfig
from recursive_feedback_loop.roundtable import (
    RoundTableConfig,
    RoundTableRunner,
    ROUNDTABLE_SYNTHESIS,
)
from recursive_feedback_loop.session_reader import Turn, Conversation
from recursive_feedback_loop.loop_runner import LoopState


# --- RoundTableConfig ---

class TestRoundTableConfig:
    def _make_config(self, **kwargs):
        defaults = {"agents": [AgentConfig(name="test")]}
        defaults.update(kwargs)
        return RoundTableConfig(**defaults)

    def test_default_synthesis(self):
        c = self._make_config()
        resolved = c.resolve_synthesis_instruction()
        assert resolved == ROUNDTABLE_SYNTHESIS
        assert "roundtable" in resolved.lower()

    def test_custom_synthesis(self):
        c = self._make_config(synthesis_instruction="Custom instruction")
        assert c.resolve_synthesis_instruction() == "Custom instruction"

    def test_synthesis_file(self):
        fd, path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w") as f:
            f.write("File-based instruction")
        try:
            c = self._make_config(synthesis_instruction_file=path)
            assert c.resolve_synthesis_instruction() == "File-based instruction"
        finally:
            os.unlink(path)

    def test_none_synthesis_falls_back_to_default(self):
        c = self._make_config(synthesis_instruction=None)
        assert c.resolve_synthesis_instruction() == ROUNDTABLE_SYNTHESIS

    def test_seed_prompt_string(self):
        c = self._make_config(seed_prompt="Hello")
        assert c.resolve_seed_prompt() == "Hello"

    def test_seed_prompt_file(self):
        fd, path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w") as f:
            f.write("File seed content")
        try:
            c = self._make_config(seed_prompt_file=path)
            assert c.resolve_seed_prompt() == "File seed content"
        finally:
            os.unlink(path)

    def test_seed_prompt_missing_raises(self):
        c = self._make_config()
        c.seed_prompt = ""
        with pytest.raises(ValueError, match="No seed prompt"):
            c.resolve_seed_prompt()

    def test_output_dir_auto_timestamp(self):
        c = self._make_config()
        d = c.get_output_dir()
        assert "rfl_roundtable_" in str(d)
        # Cleanup
        import shutil
        shutil.rmtree(d, ignore_errors=True)

    def test_output_dir_custom(self):
        tmpdir = tempfile.mkdtemp()
        try:
            c = self._make_config(output_dir=tmpdir)
            assert c.get_output_dir() == Path(tmpdir)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_run_id_auto(self):
        c = self._make_config()
        rid = c.get_run_id()
        assert rid.startswith("roundtable_")

    def test_run_id_custom(self):
        c = self._make_config(run_id="my_custom_id")
        assert c.get_run_id() == "my_custom_id"

    def test_seed_timeout_default(self):
        c = self._make_config(iteration_timeout=300)
        assert c._seed_timeout() == 600

    def test_seed_timeout_custom(self):
        c = self._make_config(seed_timeout=120, iteration_timeout=300)
        assert c._seed_timeout() == 120


# --- RoundTableRunner._build_agent_prompt ---

class TestBuildAgentPrompt:
    def _make_runner(self, agents=None, **kwargs):
        if agents is None:
            agents = [
                AgentConfig(name="alpha", role="Find bugs"),
                AgentConfig(name="beta", role="Find perf"),
            ]
        config = RoundTableConfig(
            agents=agents,
            seed_prompt="test seed",
            max_rounds=3,
            **kwargs,
        )
        return RoundTableRunner(config)

    def test_seed_prompt_contains_role(self):
        runner = self._make_runner()
        agent = AgentConfig(name="auditor", role="Find security bugs")
        prompt = runner._build_agent_prompt(agent, "Do stuff", ROUNDTABLE_SYNTHESIS, is_seed=True)
        assert "Find security bugs" in prompt
        assert "auditor" in prompt

    def test_seed_prompt_says_initial_task(self):
        runner = self._make_runner()
        agent = runner.config.agents[0]
        prompt = runner._build_agent_prompt(agent, "seed context", ROUNDTABLE_SYNTHESIS, is_seed=True)
        assert "INITIAL TASK" in prompt
        # Should NOT say "This is round N of M" -- that's for non-seed
        assert "This is round" not in prompt

    def test_round_prompt_contains_history(self):
        runner = self._make_runner()
        agent = runner.config.agents[0]
        prompt = runner._build_agent_prompt(
            agent, "compacted history here", ROUNDTABLE_SYNTHESIS,
            is_seed=False, round_num=1,
        )
        assert "SHARED CONVERSATION HISTORY" in prompt
        assert "compacted history here" in prompt
        assert "round 1 of 3" in prompt

    def test_round_prompt_has_go_deeper_instruction(self):
        runner = self._make_runner()
        agent = runner.config.agents[0]
        prompt = runner._build_agent_prompt(
            agent, "ctx", ROUNDTABLE_SYNTHESIS, is_seed=False, round_num=2,
        )
        assert "round 2 of 3" in prompt
        assert "Do NOT repeat" in prompt

    def test_no_role_no_crash(self):
        runner = self._make_runner()
        agent = AgentConfig(name="bare")
        prompt = runner._build_agent_prompt(agent, "ctx", ROUNDTABLE_SYNTHESIS, is_seed=True)
        assert "YOUR ROLE" not in prompt
        assert "bare" in prompt  # name still shown


# --- RoundTableRunner._export_markdown ---

class TestExportMarkdown:
    def test_agent_tags_in_headers(self):
        tmpdir = Path(tempfile.mkdtemp())
        config = RoundTableConfig(
            agents=[
                AgentConfig(name="arch", model="m1"),
                AgentConfig(name="perf", model="m2"),
            ],
            seed_prompt="test",
        )
        runner = RoundTableRunner(config)
        runner.state = LoopState(tmpdir)
        runner.state.conversation.add("user", "seed", iteration=0)
        runner.state.conversation.add("assistant", "arch response", iteration=0, agent="arch")
        runner.state.conversation.add("assistant", "perf response", iteration=0, agent="perf")

        path = runner._export_markdown()
        content = path.read_text()

        assert "[arch]" in content
        assert "[perf]" in content
        assert "Round 0" in content
        # User turn should NOT have agent tag
        assert "## USER (Round 0)" in content

        import shutil
        shutil.rmtree(runner.state.output_dir, ignore_errors=True)


# --- Lock file ---

class TestLockFile:
    def test_acquire_and_release(self):
        tmpdir = tempfile.mkdtemp()
        config = RoundTableConfig(
            agents=[AgentConfig(name="test")],
            seed_prompt="test",
            output_dir=tmpdir,
        )
        runner = RoundTableRunner(config)
        runner.state = LoopState(Path(tmpdir))

        assert runner._acquire_lock()
        lockpath = config.get_lockfile_path()
        assert lockpath.exists()

        data = json.loads(lockpath.read_text())
        assert data["mode"] == "roundtable"
        assert "test" in data["agents"]

        runner._release_lock()
        assert not lockpath.exists()

        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_double_acquire_fails(self):
        tmpdir = tempfile.mkdtemp()
        config = RoundTableConfig(
            agents=[AgentConfig(name="test")],
            seed_prompt="test",
            output_dir=tmpdir,
        )
        runner1 = RoundTableRunner(config)
        runner1.state = LoopState(Path(tmpdir))
        runner1._acquire_lock()

        runner2 = RoundTableRunner(config)
        runner2.state = LoopState(Path(tmpdir))
        assert not runner2._acquire_lock()  # locked by runner1's PID

        runner1._release_lock()

        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# --- Hermes query mock ---

class TestRunAgentQuery:
    def test_successful_query(self):
        config = RoundTableConfig(
            agents=[AgentConfig(name="test", model="fake-model")],
            seed_prompt="test",
        )
        runner = RoundTableRunner(config)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "This is the agent response"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            response = runner._run_agent_query(
                AgentConfig(name="test", model="fake-model"),
                "test prompt",
            )
            assert response == "This is the agent response"
            # Verify correct command construction
            cmd = mock_run.call_args[0][0]
            assert "hermes" in cmd[0] or cmd[0] == "hermes"
            assert "-m" in cmd
            assert "fake-model" in cmd

    def test_failed_query_returns_none(self):
        tmpdir = Path(tempfile.mkdtemp())
        config = RoundTableConfig(
            agents=[AgentConfig(name="test")],
            seed_prompt="test",
        )
        runner = RoundTableRunner(config)
        runner.state = LoopState(tmpdir)

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "error message"

        with patch("subprocess.run", return_value=mock_result):
            response = runner._run_agent_query(
                AgentConfig(name="test"), "prompt"
            )
            assert response is None

        import shutil
        shutil.rmtree(runner.state.output_dir, ignore_errors=True)

    def test_timeout_returns_none(self):
        import subprocess
        tmpdir = Path(tempfile.mkdtemp())
        config = RoundTableConfig(
            agents=[AgentConfig(name="test")],
            seed_prompt="test",
        )
        runner = RoundTableRunner(config)
        runner.state = LoopState(tmpdir)

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 60)):
            response = runner._run_agent_query(
                AgentConfig(name="test"), "prompt"
            )
            assert response is None

        import shutil
        shutil.rmtree(runner.state.output_dir, ignore_errors=True)
