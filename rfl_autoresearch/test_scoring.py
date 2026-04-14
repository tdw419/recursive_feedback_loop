"""Tests for RFL autoresearch scoring functions."""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rfl_autoresearch.experiment import (
    score_delta_ratio,
    score_specificity,
    score_depth_progression,
    score_dedup_efficiency,
    compute_composite,
    _count_specificity_markers,
    _word_count,
)


def test_word_count():
    assert _word_count("hello world") == 2
    assert _word_count("") == 0
    assert _word_count("one") == 1


def test_specificity_markers():
    text = "Call `foo()` at line 42 in file.py"
    markers = _count_specificity_markers(text)
    assert markers >= 3  # foo(), 42, file.py at minimum


def test_delta_ratio_identical():
    """Identical turns should have low delta."""
    turns = ["The quick brown fox jumps over the lazy dog"] * 3
    ratio = score_delta_ratio(turns)
    # With stopword filtering, the content words are quick/brown/fox/jumps/lazy/dog
    # Since they're identical, delta should be very low
    assert ratio < 0.3


def test_delta_ratio_different():
    """Completely different turns should have high delta."""
    turns = [
        "alpha beta gamma delta epsilon",
        "zeta eta theta iota kappa",
        "lambda mu nu xi omicron",
    ]
    ratio = score_delta_ratio(turns)
    assert ratio > 0.9  # no shared content words


def test_delta_ratio_single():
    """Single turn should return 1.0."""
    ratio = score_delta_ratio(["hello world"])
    assert ratio == 1.0


def test_delta_ratio_empty():
    ratio = score_delta_ratio([])
    assert ratio == 1.0


def test_specificity_concrete():
    """Text with numbers, code, paths should score higher."""
    concrete = "Bug at line 42: `foo(bar)` returns None. Fix in utils.py:\n```python\nreturn x\n```"
    vague = "There might be some issue with the code somewhere that could be improved."
    assert score_specificity(concrete) > score_specificity(vague)


def test_depth_progression_increasing():
    """Turns that get more specific should score high."""
    turns = [
        "The code has some issues.",
        "Bug found: `parse_config()` crashes on line 42 with KeyError.",
        "Root cause: `parse_config()` at line 42 calls `data.get('settings', {})` but "
        "the JSON has `config` not `settings`. Fix: change to `data.get('config', {})`. "
        "Also `run_cmd()` at line 8 uses `shell=True` which is a shell injection risk.",
    ]
    score = score_depth_progression(turns)
    assert score > 0.3  # should show progression


def test_depth_progression_flat():
    """Identical specificity should be neutral."""
    turns = [
        "Some text with 1 reference.",
        "Other text with 1 reference.",
    ]
    score = score_depth_progression(turns)
    # Should be near 0.5 (neutral)
    assert 0.3 < score < 0.7


def test_dedup_efficiency():
    """Duplicated text should show high efficiency."""
    text = "Hello world this is a test. " * 5
    raw = [text]
    deduped = [text[:len(text)//2]]  # simulate dedup
    eff = score_dedup_efficiency(raw, deduped)
    assert eff > 0.3  # caught significant duplication


def test_dedup_no_change():
    """No change means 0 efficiency."""
    text = "unique content here"
    eff = score_dedup_efficiency([text], [text])
    assert eff == 0.0


def test_composite_range():
    """Composite should be between 0 and 1."""
    score = compute_composite(0.5, 5.0, 0.5, 0.1)
    assert 0.0 <= score <= 1.0


def test_composite_high():
    """Good metrics should produce higher composite."""
    high = compute_composite(0.8, 8.0, 0.7, 0.2)
    low = compute_composite(0.2, 1.0, 0.2, 0.0)
    assert high > low


def test_specificity_empty():
    assert score_specificity("") == 0.0
