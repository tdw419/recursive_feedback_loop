"""Compaction strategies for managing context growth in recursive loops.

The core problem: each iteration produces output that feeds the next iteration.
Without compaction, context grows exponentially: O(response_size * 2^n).

Solutions:
  - SlidingWindow: Keep last N turns verbatim, discard rest. Simple, loses history.
  - RollingSummary: LLM-summarize old turns into a compact state. Preserves themes.
  - Hierarchical: Recent turns verbatim, medium turns as bullets, old turns as summary.
    Best balance of detail and context budget.
"""

import subprocess
import json
import sys
from abc import ABC, abstractmethod
from typing import List, Optional

from .session_reader import Turn, Conversation


class CompactionStrategy(ABC):
    """Base class for compaction strategies."""

    @abstractmethod
    def compact(self, conversation: Conversation, token_budget: int) -> str:
        """Compact a conversation into a string that fits within token_budget.

        Returns a string representation of the compacted context.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        ...


class SlidingWindow(CompactionStrategy):
    """Keep only the last N turns. Oldest turns are dropped entirely."""

    def __init__(self, keep_turns: int = 4):
        self.keep_turns = keep_turns

    def name(self) -> str:
        return "sliding_window"

    def compact(self, conversation: Conversation, token_budget: int) -> str:
        turns = conversation.last_n_turns(self.keep_turns)
        return _format_turns(turns, token_budget)


class RollingSummary(CompactionStrategy):
    """Maintain a running summary of old turns, keep recent ones verbatim.

    Uses an LLM call to summarize when context exceeds budget.
    Falls back to truncation if no LLM available.
    """

    def __init__(
        self,
        recent_turns: int = 3,
        summary_max_chars: int = 500,
        compaction_model: Optional[str] = None,
        compaction_provider: Optional[str] = None,
    ):
        self.recent_turns = recent_turns
        self.summary_max_chars = summary_max_chars
        self.compaction_model = compaction_model
        self.compaction_provider = compaction_provider
        self._running_summary = ""
        self._last_summarized_idx = -1  # track what's been summarized

    def name(self) -> str:
        return "rolling_summary"

    def compact(self, conversation: Conversation, token_budget: int) -> str:
        recent = conversation.last_n_turns(self.recent_turns)
        # Only summarize turns not yet incorporated into the summary
        older = conversation.turns[:max(0, len(conversation.turns) - self.recent_turns)]
        new_older = older[self._last_summarized_idx + 1:]

        if new_older:
            new_text = _format_turns(new_older, token_budget * 2)
            new_summary = self._summarize(new_text)
            if new_summary:
                if self._running_summary:
                    self._running_summary = (
                        self._running_summary + " " + new_summary
                    )
                    if len(self._running_summary) > self.summary_max_chars:
                        # Re-compress when exceeding budget
                        self._running_summary = self._summarize(self._running_summary) or self._running_summary[:self.summary_max_chars]
                else:
                    self._running_summary = new_summary
            self._last_summarized_idx = len(older) - 1

        parts = []
        if self._running_summary:
            parts.append(f"[PREVIOUS CONTEXT SUMMARY]\n{self._running_summary}\n")
        parts.append("[RECENT TURNS]")
        parts.append(_format_turns(recent, token_budget))

        result = "\n".join(parts)
        # Hard truncate if still over budget
        return _truncate_to_tokens(result, token_budget)

    def _summarize(self, text: str) -> Optional[str]:
        """Use Hermes to summarize text. Falls back to extractive summary."""
        if self.compaction_model or self.compaction_provider:
            return self._llm_summarize(text)
        # Fallback: extractive summary (first sentences of paragraphs)
        return self._extractive_summary(text)

    def _llm_summarize(self, text: str) -> Optional[str]:
        """Summarize using Hermes CLI."""
        prompt = (
            f"Summarize the following conversation turns in {self.summary_max_chars} chars or less. "
            f"Capture the key ideas, conclusions, and unresolved questions. Be dense and informative.\n\n"
            f"{text}"
        )
        cmd = ["hermes", "chat", "-q", prompt, "-Q", "-t", ""]
        if self.compaction_model:
            cmd.extend(["-m", self.compaction_model])
        if self.compaction_provider:
            cmd.extend(["--provider", self.compaction_provider])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0 and result.stdout.strip():
                summary = result.stdout.strip()
                if len(summary) > self.summary_max_chars:
                    summary = summary[: self.summary_max_chars - 3] + "..."
                return summary
        except (subprocess.TimeoutExpired, Exception):
            pass
        return None

    def _extractive_summary(self, text: str) -> str:
        """Simple extractive summary: first sentence of each paragraph."""
        sentences = []
        for para in text.split("\n"):
            para = para.strip()
            if not para or para.startswith("[") or para.startswith("Turn"):
                continue
            # Get first sentence (rough)
            dot_pos = para.find(". ")
            if dot_pos > 0 and dot_pos < 200:
                sentences.append(para[: dot_pos + 1])
            elif len(para) > 200:
                sentences.append(para[:197] + "...")
            else:
                sentences.append(para)
            if len(sentences) >= 5:
                break

        summary = " ".join(sentences)
        if len(summary) > self.summary_max_chars:
            summary = summary[: self.summary_max_chars - 3] + "..."
        return summary


class Hierarchical(CompactionStrategy):
    """Three-tier context: old = summary, medium = bullets, recent = verbatim.

    This is the default strategy. Best balance of preserving context
    and staying within token budget.
    """

    def __init__(
        self,
        recent_turns: int = 3,
        medium_turns: int = 5,
        summary_max_chars: int = 500,
        compaction_model: Optional[str] = None,
        compaction_provider: Optional[str] = None,
    ):
        self.recent_turns = recent_turns
        self.medium_turns = medium_turns
        self.summary_max_chars = summary_max_chars
        self.compaction_model = compaction_model
        self.compaction_provider = compaction_provider
        self._accumulated_summary = ""
        self._last_old_idx = -1  # track what's been summarized

    def name(self) -> str:
        return "hierarchical"

    def compact(self, conversation: Conversation, token_budget: int) -> str:
        n = len(conversation.turns)
        recent = conversation.last_n_turns(self.recent_turns)
        medium_start = max(0, n - self.recent_turns - self.medium_turns)
        medium = conversation.turns[medium_start : n - self.recent_turns] if n > self.recent_turns else []
        old = conversation.turns[:medium_start] if medium_start > 0 else []

        # Update accumulated summary with old turns (only new ones)
        if old:
            new_old = old[self._last_old_idx + 1:]
            if new_old:
                old_text = _format_turns(new_old, token_budget * 2)
                new_summary = self._summarize(old_text)
                if new_summary:
                    if self._accumulated_summary:
                        self._accumulated_summary = self._summarize(
                            f"Previous summary: {self._accumulated_summary}\n\nNew content to integrate: {new_summary}"
                        ) or (self._accumulated_summary + " " + new_summary)
                    else:
                        self._accumulated_summary = new_summary
                    # Keep summary bounded
                    if len(self._accumulated_summary) > self.summary_max_chars:
                        self._accumulated_summary = self._accumulated_summary[: self.summary_max_chars - 3] + "..."
                self._last_old_idx = len(old) - 1

        parts = []

        # Tier 1: Old summary
        if self._accumulated_summary:
            parts.append(f"[DISTANT CONTEXT]\n{self._accumulated_summary}\n")

        # Tier 2: Medium turns as bullets
        if medium:
            parts.append("[RECENT CONTEXT (condensed)]")
            for turn in medium:
                # Extract key points as bullets
                bullets = _turn_to_bullets(turn)
                agent_label = f", agent={turn.agent}" if turn.agent else ""
                parts.append(f"  - ({turn.role}{agent_label}, round {turn.iteration}): {bullets}")
            parts.append("")

        # Tier 3: Recent turns verbatim
        parts.append("[LATEST TURNS (verbatim)]")
        parts.append(_format_turns(recent, token_budget))

        result = "\n".join(parts)
        return _truncate_to_tokens(result, token_budget)

    def _summarize(self, text: str) -> Optional[str]:
        """Summarize using LLM or extractive fallback."""
        if self.compaction_model or self.compaction_provider:
            prompt = (
                f"Summarize in under {self.summary_max_chars} chars. "
                f"Key ideas, conclusions, themes only. Dense and informative.\n\n{text}"
            )
            cmd = ["hermes", "chat", "-q", prompt, "-Q", "-t", ""]
            if self.compaction_model:
                cmd.extend(["-m", self.compaction_model])
            if self.compaction_provider:
                cmd.extend(["--provider", self.compaction_provider])
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0 and result.stdout.strip():
                    s = result.stdout.strip()
                    return s[: self.summary_max_chars] if len(s) > self.summary_max_chars else s
            except Exception:
                pass

        # Fallback: extractive
        sentences = []
        for line in text.split("\n"):
            line = line.strip()
            if not line or line.startswith("[") or line.startswith("Turn"):
                continue
            for sentence in line.split(". "):
                sentence = sentence.strip()
                if sentence and len(sentence) > 20:
                    sentences.append(sentence)
                    break
            if len(sentences) >= 5:
                break
        summary = ". ".join(sentences)
        return summary[: self.summary_max_chars]


# --- Helpers ---

def _format_turns(turns: List[Turn], token_budget: int) -> str:
    """Format turns as readable text, truncating to fit budget.

    Agent-aware: if a turn has an 'agent' field, includes the agent name
    in the header. This preserves attribution through compaction.
    """
    parts = []
    for t in turns:
        agent_tag = f" by {t.agent}" if t.agent else ""
        parts.append(f"[{t.role}{agent_tag} (round {t.iteration})]")
        parts.append(t.content)
        parts.append("")
    return _truncate_to_tokens("\n".join(parts), token_budget)


def _turn_to_bullets(turn: Turn) -> str:
    """Extract key points from a turn as a condensed bullet string.

    Agent-aware: prepends agent name if present.
    """
    agent_tag = f"[{turn.agent}] " if turn.agent else ""
    content = turn.content
    # Split into sentences using regex to avoid mangling != and file extensions
    import re
    sentences = []
    # Split on sentence boundaries: period/question/excl followed by space and uppercase
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', content)
    for s in parts:
        s = s.strip()
        if s and len(s) > 15:
            sentences.append(s)
        if len(sentences) >= 3:
            break
    if not sentences:
        return agent_tag + content[:150] + ("..." if len(content) > 150 else "")
    return agent_tag + "; ".join(sentences)


def _truncate_to_tokens(text: str, token_budget: int) -> str:
    """Truncate text to fit within a rough token budget (chars / 4)."""
    char_budget = token_budget * 4
    if len(text) <= char_budget:
        return text
    return text[: char_budget - 50] + "\n\n[... truncated to fit token budget ...]"


def get_strategy(name: str, **kwargs) -> CompactionStrategy:
    """Factory function to get a compaction strategy by name."""
    strategies = {
        "sliding_window": SlidingWindow,
        "rolling_summary": RollingSummary,
        "hierarchical": Hierarchical,
    }
    cls = strategies.get(name)
    if not cls:
        raise ValueError(f"Unknown compaction strategy: {name}. Choose from: {list(strategies.keys())}")
    return cls(**kwargs)
