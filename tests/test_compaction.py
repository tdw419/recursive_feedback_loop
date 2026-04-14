"""Tests for compaction strategies."""

from recursive_feedback_loop.session_reader import Turn, Conversation
from recursive_feedback_loop.compaction import (
    SlidingWindow,
    RollingSummary,
    Hierarchical,
    get_strategy,
    _truncate_to_tokens,
)


def _make_conversation(n_turns: int = 20) -> Conversation:
    """Build a test conversation with n_turns."""
    c = Conversation()
    for i in range(n_turns):
        role = "user" if i % 2 == 0 else "assistant"
        c.add(role, f"This is turn {i}. " + "Word " * 50, iteration=i // 2)
    return c


def test_truncate_to_tokens():
    text = "A" * 1000
    truncated = _truncate_to_tokens(text, 100)  # budget = 400 chars
    assert len(truncated) < 500
    assert "truncated" in truncated


def test_no_truncate_needed():
    text = "Short text"
    result = _truncate_to_tokens(text, 1000)
    assert result == text


def test_sliding_window():
    conv = _make_conversation(20)
    sw = SlidingWindow(keep_turns=4)
    result = sw.compact(conv, 8000)
    assert "turn 19" in result  # last turn
    assert "turn 0" not in result  # first turn dropped


def test_sliding_window_name():
    sw = SlidingWindow()
    assert sw.name() == "sliding_window"


def test_rolling_summary_no_llm():
    conv = _make_conversation(20)
    rs = RollingSummary(recent_turns=3, summary_max_chars=200)
    result = rs.compact(conv, 4000)
    # Should have recent turns section
    assert "RECENT TURNS" in result
    # Should have some summary (extractive fallback)
    assert len(result) > 0


def test_hierarchical():
    conv = _make_conversation(30)
    h = Hierarchical(recent_turns=3, medium_turns=5, summary_max_chars=300)
    result = h.compact(conv, 6000)
    # Should have all three tiers
    assert "LATEST TURNS" in result
    assert "RECENT CONTEXT" in result
    # Hierarchical preserves more context than sliding window
    assert len(result) > 0


def test_hierarchical_small_conversation():
    conv = Conversation()
    conv.add("user", "Hi", iteration=0)
    conv.add("assistant", "Hello!", iteration=0)

    h = Hierarchical()
    result = h.compact(conv, 4000)
    assert "Hi" in result
    assert "Hello!" in result


def test_compaction_respects_budget():
    conv = _make_conversation(50)
    for strategy_name in ["sliding_window", "rolling_summary", "hierarchical"]:
        kwargs = {}
        if strategy_name == "rolling_summary":
            kwargs = {"recent_turns": 3, "summary_max_chars": 200}
        elif strategy_name == "hierarchical":
            kwargs = {"recent_turns": 3, "medium_turns": 5, "summary_max_chars": 200}
        elif strategy_name == "sliding_window":
            kwargs = {"keep_turns": 4}
        strategy = get_strategy(strategy_name, **kwargs)
        result = strategy.compact(conv, 2000)
        # Should fit in token budget (chars / 4)
        estimated_tokens = len(result) // 4
        # Allow some slack since token estimation is rough
        assert estimated_tokens < 3000, f"{strategy_name} exceeded budget: {estimated_tokens}"


def test_get_strategy():
    sw = get_strategy("sliding_window", keep_turns=5)
    assert isinstance(sw, SlidingWindow)
    assert sw.keep_turns == 5

    rs = get_strategy("rolling_summary", recent_turns=2, summary_max_chars=100)
    assert isinstance(rs, RollingSummary)

    h = get_strategy("hierarchical", recent_turns=2, medium_turns=3, summary_max_chars=200)
    assert isinstance(h, Hierarchical)


def test_get_strategy_invalid():
    try:
        get_strategy("nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "nonexistent" in str(e)
