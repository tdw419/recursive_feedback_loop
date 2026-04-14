"""Read and parse Hermes session JSONL files."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class Turn:
    """A single conversation turn."""
    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: Optional[str] = None
    iteration: int = 0  # which loop iteration this belongs to

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "iteration": self.iteration,
        }

    def token_estimate(self) -> int:
        """Rough token estimate (chars / 4)."""
        return len(self.content) // 4

    def __repr__(self):
        preview = self.content[:80].replace("\n", "\\n")
        return f"Turn(role={self.role!r}, iter={self.iteration}, tokens~{self.token_estimate()}, preview={preview!r})"


@dataclass
class Conversation:
    """A sequence of turns."""
    turns: List[Turn] = field(default_factory=list)

    def add(self, role: str, content: str, timestamp: str = None, iteration: int = 0):
        self.turns.append(Turn(role=role, content=content, timestamp=timestamp, iteration=iteration))

    def total_tokens_estimate(self) -> int:
        return sum(t.token_estimate() for t in self.turns)

    def last_n_turns(self, n: int) -> List[Turn]:
        return self.turns[-n:] if n > 0 else []

    def turns_by_iteration(self, iteration: int) -> List[Turn]:
        return [t for t in self.turns if t.iteration == iteration]

    def to_dicts(self) -> List[dict]:
        return [t.to_dict() for t in self.turns]

    def __len__(self):
        return len(self.turns)

    def __repr__(self):
        return f"Conversation({len(self.turns)} turns, ~{self.total_tokens_estimate()} tokens)"


class SessionReader:
    """Reads Hermes JSONL session files and parses them into Conversations."""

    def __init__(self, session_path: str):
        self.session_path = Path(session_path)
        self._last_line = 0

    @staticmethod
    def find_latest_session(sessions_dir: str = None) -> Optional[Path]:
        """Find the most recent session file."""
        if sessions_dir is None:
            sessions_dir = Path.home() / ".hermes" / "sessions"
        else:
            sessions_dir = Path(sessions_dir)

        session_files = sorted(sessions_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        return session_files[-1] if session_files else None

    def read_new_turns(self) -> List[Turn]:
        """Read turns that have been appended since the last call."""
        new_turns = []
        if not self.session_path.exists():
            return new_turns

        with open(self.session_path, "r") as f:
            for i, line in enumerate(f):
                if i < self._last_line:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Filter out tool calls and system messages for clean conversation
                    role = data.get("role", "unknown")
                    content = data.get("content", "")
                    if role in ("user", "assistant") and content:
                        ts = data.get("timestamp")
                        new_turns.append(Turn(role=role, content=content, timestamp=ts))
                except json.JSONDecodeError:
                    continue
            self._last_line = i + 1

        return new_turns

    def read_all(self) -> Conversation:
        """Read the entire session file."""
        self._last_line = 0
        conv = Conversation()
        turns = self.read_new_turns()
        for t in turns:
            conv.turns.append(t)
        return conv


def extract_assistant_response(turns: List[Turn]) -> Optional[str]:
    """Get the last assistant response from a list of new turns."""
    for turn in reversed(turns):
        if turn.role == "assistant":
            return turn.content
    return None
