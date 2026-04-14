"""Tests for config."""

from recursive_feedback_loop.config import LoopConfig
import tempfile
from pathlib import Path


def test_default_config():
    config = LoopConfig()
    assert config.max_iterations == 10
    assert config.compaction_strategy == "hierarchical"
    assert config.max_context_tokens == 8000


def test_resolve_seed_prompt():
    config = LoopConfig(seed_prompt="Hello world")
    assert config.resolve_seed_prompt() == "Hello world"


def test_resolve_seed_from_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("File-based seed prompt")
        path = f.name
    config = LoopConfig(seed_prompt_file=path)
    assert config.resolve_seed_prompt() == "File-based seed prompt"


def test_resolve_seed_prompt_missing():
    config = LoopConfig()
    try:
        config.resolve_seed_prompt()
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_output_dir_creation():
    with tempfile.TemporaryDirectory() as td:
        config = LoopConfig(output_dir=str(Path(td) / "new_dir"))
        out = config.get_output_dir()
        assert out.exists()
        assert out.name == "new_dir"


def test_synthesis_instruction():
    config = LoopConfig(synthesis_instruction="Go deeper")
    assert config.resolve_synthesis_instruction() == "Go deeper"


def test_synthesis_from_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Custom synthesis instruction")
        path = f.name
    config = LoopConfig(synthesis_instruction_file=path)
    assert config.resolve_synthesis_instruction() == "Custom synthesis instruction"
