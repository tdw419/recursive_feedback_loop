"""Tests for agents.py -- AgentConfig, parsing, validation."""

import pytest
import tempfile
import os

from recursive_feedback_loop.agents import (
    AgentConfig,
    parse_agent_string,
    load_agents_file,
    validate_agents,
)


# --- AgentConfig ---

class TestAgentConfig:
    def test_display_with_model_and_provider(self):
        a = AgentConfig(name="auditor", model="claude-sonnet-4", provider="anthropic")
        assert a.display() == "auditor claude-sonnet-4 via anthropic"

    def test_display_name_only(self):
        a = AgentConfig(name="test")
        assert a.display() == "test"

    def test_display_model_no_provider(self):
        a = AgentConfig(name="x", model="gpt-4")
        assert a.display() == "x gpt-4"


# --- parse_agent_string ---

class TestParseAgentString:
    def test_valid_full(self):
        a = parse_agent_string("name=foo,model=bar,provider=baz,role=some role")
        assert a.name == "foo"
        assert a.model == "bar"
        assert a.provider == "baz"
        assert a.role == "some role"

    def test_name_only(self):
        a = parse_agent_string("name=only_name")
        assert a.name == "only_name"
        assert a.model is None

    def test_missing_name_raises(self):
        with pytest.raises(ValueError, match="requires a 'name'"):
            parse_agent_string("model=foo,role=bar")

    def test_quoted_values(self):
        a = parse_agent_string('name=foo,role="a role with spaces"')
        assert a.role == "a role with spaces"

    def test_single_quotes(self):
        a = parse_agent_string("name=foo,role='quoted role'")
        assert a.role == "quoted role"


# --- load_agents_file ---

class TestLoadAgentsFile:
    def _write_yaml(self, content):
        fd, path = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, "w") as f:
            f.write(content)
        return path

    def test_yaml_block_format(self):
        path = self._write_yaml(
            "agents:\n"
            "  - name: auditor\n"
            "    model: claude-sonnet-4\n"
            "    provider: anthropic\n"
            "    role: Find bugs\n"
            "  - name: optimizer\n"
            "    model: gemini-pro\n"
            "    role: Find perf issues\n"
        )
        try:
            agents = load_agents_file(path)
            assert len(agents) == 2
            assert agents[0].name == "auditor"
            assert agents[0].model == "claude-sonnet-4"
            assert agents[1].name == "optimizer"
        finally:
            os.unlink(path)

    def test_inline_format(self):
        path = self._write_yaml(
            "agents:\n"
            "  - name=foo,model=bar\n"
            "  - name=baz,role=some role\n"
        )
        try:
            agents = load_agents_file(path)
            assert len(agents) == 2
            assert agents[0].name == "foo"
            assert agents[1].role == "some role"
        finally:
            os.unlink(path)

    def test_empty_agents_raises(self):
        path = self._write_yaml("agents:\n")
        try:
            with pytest.raises(ValueError, match="No agents found"):
                load_agents_file(path)
        finally:
            os.unlink(path)

    def test_no_agents_key_raises(self):
        path = self._write_yaml("something_else: true\n")
        try:
            with pytest.raises(ValueError, match="No agents found"):
                load_agents_file(path)
        finally:
            os.unlink(path)


# --- validate_agents ---

class TestValidateAgents:
    def test_valid_agents_no_warnings(self):
        agents = [
            AgentConfig(name="a", model="m1", role="r1"),
            AgentConfig(name="b", model="m2", role="r2"),
        ]
        assert validate_agents(agents) == []

    def test_duplicate_names_warns(self):
        agents = [
            AgentConfig(name="same", model="m1", role="r1"),
            AgentConfig(name="same", model="m2", role="r2"),
        ]
        warnings = validate_agents(agents)
        assert any("Duplicate" in w for w in warnings)

    def test_no_model_no_provider_warns(self):
        agents = [AgentConfig(name="bare", role="has role")]
        warnings = validate_agents(agents)
        assert any("no model or provider" in w for w in warnings)

    def test_no_role_warns(self):
        agents = [AgentConfig(name="x", model="m")]
        warnings = validate_agents(agents)
        assert any("no role" in w for w in warnings)
