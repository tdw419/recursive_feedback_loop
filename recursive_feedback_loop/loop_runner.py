"""Core loop runner — orchestrates the recursive feedback loop.

Flow:
  1. Seed prompt → Hermes session
  2. Poll session file for AI response
  3. Compact conversation history
  4. Build synthesis prompt from compacted context
  5. Feed synthesis prompt back into Hermes
  6. Repeat until max iterations or timeout
"""

import json
import time
import subprocess
import signal
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from .config import LoopConfig
from .session_reader import SessionReader, Turn, Conversation, extract_assistant_response
from .compaction import get_strategy, CompactionStrategy
from .session_mode import HermesSession


def _deduplicate_response(text: str) -> str:
    """Hermes sometimes outputs the same content twice (boxed + final).
    Detect and remove the duplicate if the first half equals the second half.
    """
    mid = len(text) // 2
    if mid < 200:
        return text

    # Strategy 1: Check if second half starts with same content as first
    first_half = text[:mid].strip()
    second_half = text[mid:].strip()
    overlap_len = min(len(first_half), len(second_half), 200)
    if overlap_len > 50:
        first_prefix = first_half[:overlap_len]
        second_prefix = second_half[:overlap_len]
        # Allow some whitespace differences
        if first_prefix.replace(" ", "") == second_prefix.replace(" ", ""):
            return first_half

    # Strategy 2: Find the longest repeated block anywhere
    # Use a sliding window approach
    for chunk_size in [800, 500, 300, 200, 100]:
        if chunk_size > len(text) // 3:
            continue
        chunk = text[:chunk_size]
        # Look for this chunk appearing again later
        search_start = chunk_size + 50
        idx = text.find(chunk, search_start)
        if idx > search_start:
            # Check that the duplicate is substantial
            deduped = text[:idx].strip()
            if len(deduped) > chunk_size:
                return deduped

    # Strategy 3: Find any line that appears twice as a split point
    lines = text.split("\n")
    seen_lines = {}
    for i, line in enumerate(lines):
        stripped = line.strip()
        if len(stripped) > 50:  # Only substantial lines
            if stripped in seen_lines and i > len(lines) // 3:
                # Found a duplicate line — take everything before it
                deduped = "\n".join(lines[:i]).strip()
                if len(deduped) > len(text) // 3:
                    return deduped
            seen_lines[stripped] = i

    return text


class LoopState:
    """Tracks the state of a running loop."""

    def __init__(self, output_dir: Path):
        self.iteration = 0
        self.conversation = Conversation()
        self.start_time = time.time()
        self.session_path: Optional[Path] = None
        self.session_reader: Optional[SessionReader] = None
        self.output_dir = output_dir
        self.log_file = output_dir / "loop_log.jsonl"
        self.snapshots_dir = output_dir / "snapshots"
        self.snapshots_dir.mkdir(exist_ok=True)

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def log(self, event: str, data: dict = None):
        """Append a log entry."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "iteration": self.iteration,
            "elapsed_seconds": round(self.elapsed(), 1),
            "event": event,
        }
        if data:
            entry["data"] = data
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def save_snapshot(self):
        """Save current conversation state."""
        snap = {
            "iteration": self.iteration,
            "elapsed": self.elapsed(),
            "turns": self.conversation.to_dicts(),
            "total_tokens_estimate": self.conversation.total_tokens_estimate(),
        }
        path = self.snapshots_dir / f"iter_{self.iteration:03d}.json"
        path.write_text(json.dumps(snap, indent=2))
        return path

    def export_markdown(self) -> Path:
        """Export full conversation as markdown."""
        path = self.output_dir / "conversation.md"
        lines = [f"# Recursive Feedback Loop — Conversation Export", f"", f"Generated: {datetime.now().isoformat()}", f""]
        for turn in self.conversation.turns:
            role = turn.role.upper()
            lines.append(f"## {role} (Iteration {turn.iteration})")
            lines.append("")
            lines.append(turn.content)
            lines.append("")
            lines.append("---")
            lines.append("")
        path.write_text("\n".join(lines))
        return path

    def export_jsonl(self) -> Path:
        """Export full conversation as JSONL."""
        path = self.output_dir / "conversation.jsonl"
        with open(path, "w") as f:
            for turn in self.conversation.turns:
                f.write(json.dumps(turn.to_dict()) + "\n")
        return path


class LoopRunner:
    """Runs the recursive feedback loop."""

    def __init__(self, config: LoopConfig):
        self.config = config
        self.state: Optional[LoopState] = None
        self.compactor: Optional[CompactionStrategy] = None
        self._hermes_process = None
        self._running = False

    def run(self) -> LoopState:
        """Execute the full recursive feedback loop."""
        self._running = True
        output_dir = self.config.get_output_dir()
        self.state = LoopState(output_dir)

        # Initialize compactor
        compactor_kwargs = {}
        if self.config.compaction_strategy in ("rolling_summary", "hierarchical"):
            compactor_kwargs.update({
                "recent_turns": self.config.recent_turns_verbatim,
                "summary_max_chars": self.config.summary_max_chars,
                "compaction_model": self.config.compaction_model,
                "compaction_provider": self.config.compaction_provider,
            })
        if self.config.compaction_strategy == "hierarchical":
            compactor_kwargs["medium_turns"] = self.config.medium_turns_bullets
        elif self.config.compaction_strategy == "sliding_window":
            compactor_kwargs["keep_turns"] = self.config.recent_turns_verbatim * 2

        self.compactor = get_strategy(self.config.compaction_strategy, **compactor_kwargs)

        self.state.log("loop_start", {
            "config": {
                "max_iterations": self.config.max_iterations,
                "compaction": self.config.compaction_strategy,
                "token_budget": self.config.max_context_tokens,
                "mode": self.config.mode,
            }
        })

        # Choose execution path based on mode
        if self.config.mode == "session":
            return self._run_session_mode()
        else:
            return self._run_oneshot_mode()

    def _run_session_mode(self) -> LoopState:
        """Run the loop with a persistent Hermes tmux session.

        In session mode, Hermes keeps its full conversation history.
        We send a seed prompt, then short nudge prompts for each iteration.
        No need to feed compacted context back — Hermes already has it.
        """
        session = HermesSession(
            session_name=self.config.tmux_session_name,
            model=self.config.hermes_model,
            provider=self.config.hermes_provider,
            profile=self.config.hermes_profile,
            workdir=self.config.hermes_workdir,
        )

        self.state.log("session_starting", {"session": self.config.tmux_session_name})

        if not session.start():
            self.state.log("session_start_failed")
            self._running = False
            return self.state

        self.state.log("session_ready")

        try:
            # --- Iteration 0: Seed prompt ---
            seed = self.config.resolve_seed_prompt()
            self.state.log("seed_prompt", {"content_preview": seed[:200]})
            self.state.conversation.add("user", seed, iteration=0)

            response = session.send_prompt(seed, timeout=self.config.iteration_timeout)
            if response:
                self.state.conversation.add("assistant", response, iteration=0)
                self.state.log("seed_response", {"content_preview": response[:200], "tokens": len(response) // 4})
            else:
                self.state.log("seed_empty_response")

            # --- Iterations 1..N: Short nudge prompts ---
            instruction = self.config.resolve_synthesis_instruction()
            for i in range(1, self.config.max_iterations):
                if not self._running:
                    self.state.log("loop_stopped", {"reason": "external_stop"})
                    break

                if self.state.elapsed() > self.config.max_runtime_seconds:
                    self.state.log("loop_stopped", {"reason": "timeout"})
                    break

                self.state.iteration = i

                # In session mode, just send a short nudge — Hermes has the context
                nudge = (
                    f"{instruction}\n\n"
                    f"This is iteration {i + 1} of {self.config.max_iterations}. "
                    f"Continue from where you left off."
                )

                self.state.log("nudge_sent", {"iteration": i, "chars": len(nudge)})

                if self.config.save_snapshots:
                    snap_path = self.state.save_snapshot()
                    self.state.log("snapshot_saved", {"path": str(snap_path)})

                response = session.send_prompt(nudge, timeout=self.config.iteration_timeout)
                if response:
                    self.state.conversation.add("assistant", response, iteration=i)
                    self.state.log("iteration_response", {
                        "iteration": i,
                        "tokens": len(response) // 4,
                        "content_preview": response[:200],
                    })
                else:
                    self.state.log("iteration_empty_response", {"iteration": i})

        finally:
            session.stop()
            self.state.log("session_stopped")

        # --- Final export ---
        return self._finalize()

    def _run_oneshot_mode(self) -> LoopState:
        """Run the loop with fresh Hermes processes per iteration (original mode)."""
        seed = self.config.resolve_seed_prompt()
        self.state.log("seed_prompt", {"content_preview": seed[:200]})
        self.state.conversation.add("user", seed, iteration=0)
        response = self._run_hermes_query(seed, iteration=0)
        if response:
            self.state.conversation.add("assistant", response, iteration=0)
            self.state.log("seed_response", {"content_preview": response[:200], "tokens": len(response) // 4})

        # --- Iterations 1..N: Recursive feedback ---
        for i in range(1, self.config.max_iterations):
            if not self._running:
                self.state.log("loop_stopped", {"reason": "external_stop"})
                break

            if self.state.elapsed() > self.config.max_runtime_seconds:
                self.state.log("loop_stopped", {"reason": "timeout"})
                break

            self.state.iteration = i

            # Compact the conversation so far
            compacted = self.compactor.compact(self.state.conversation, self.config.max_context_tokens)
            self.state.log("compacted", {
                "strategy": self.compactor.name(),
                "compacted_tokens": len(compacted) // 4,
                "original_tokens": self.state.conversation.total_tokens_estimate(),
            })

            # Build the synthesis prompt
            synthesis = self._build_synthesis_prompt(compacted)
            self.state.log("synthesis_prompt", {"tokens": len(synthesis) // 4})

            # Save snapshot before next iteration
            if self.config.save_snapshots:
                snap_path = self.state.save_snapshot()
                self.state.log("snapshot_saved", {"path": str(snap_path)})

            # Feed back into Hermes
            response = self._run_hermes_query(synthesis, iteration=i)
            if response:
                self.state.conversation.add("assistant", response, iteration=i)
                self.state.log("iteration_response", {
                    "iteration": i,
                    "tokens": len(response) // 4,
                    "content_preview": response[:200],
                })
            else:
                self.state.log("iteration_empty_response", {"iteration": i})

        return self._finalize()

    def _finalize(self) -> LoopState:
        """Export results and finalize the loop state."""
        self.state.log("loop_end", {
            "total_iterations": self.state.iteration + 1,
            "total_tokens": self.state.conversation.total_tokens_estimate(),
            "elapsed_seconds": round(self.state.elapsed(), 1),
        })

        if self.config.save_snapshots:
            self.state.save_snapshot()

        if self.config.export_format in ("markdown", "both"):
            md_path = self.state.export_markdown()
            self.state.log("exported_markdown", {"path": str(md_path)})
        if self.config.export_format in ("jsonl", "both"):
            jsonl_path = self.state.export_jsonl()
            self.state.log("exported_jsonl", {"path": str(jsonl_path)})

        self._running = False
        return self.state

    def stop(self):
        """Gracefully stop the loop."""
        self._running = False

    def _run_hermes_query(self, prompt: str, iteration: int = 0) -> Optional[str]:
        """Run a single Hermes query and return the response text.

        Uses hermes chat -q for one-shot mode. Strips tool call artifacts
        from the output, keeping only the substantive text response.
        When hermes_no_tools is set, tools are disabled for iteration > 0
        (seed iteration always has tools so it can do real work).
        """
        cmd = [self.config.hermes_binary, "chat", "-q", prompt, "-Q"]
        if self.config.hermes_model:
            cmd.extend(["-m", self.config.hermes_model])
        if self.config.hermes_provider:
            cmd.extend(["--provider", self.config.hermes_provider])
        if self.config.hermes_profile:
            cmd.extend(["-p", self.config.hermes_profile])
        # Disable tools for recursive iterations (not seed) when --no-tools is set
        if self.config.hermes_no_tools and iteration > 0:
            cmd.extend(["-t", ""])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.iteration_timeout,
                cwd=self.config.hermes_workdir,
            )
            if result.returncode == 0:
                raw = result.stdout.strip()
                if raw:
                    return self._clean_hermes_output(raw)
                return None
            else:
                if self.state:
                    self.state.log("hermes_error", {
                        "returncode": result.returncode,
                        "stderr": result.stderr[:500] if result.stderr else None,
                    })
                return None
        except subprocess.TimeoutExpired:
            if self.state:
                self.state.log("hermes_timeout", {"timeout": self.config.iteration_timeout})
            return None
        except Exception as e:
            if self.state:
                self.state.log("hermes_exception", {"error": str(e)})
            return None

    @staticmethod
    def _clean_hermes_output(raw: str) -> str:
        """Strip Hermes UI artifacts from one-shot output.

        Hermes -q mode includes box-drawing characters, tool call previews,
        and other UI elements. We want just the text content.
        """
        lines = raw.split("\n")
        cleaned = []
        skip_patterns = [
            "preparing",       # tool preparation messages
            "┊",               # tool call progress lines
            "╭─",              # box drawing top
            "╰─",              # box drawing bottom
            "│",               # box drawing sides (when alone)
        ]

        for line in lines:
            stripped = line.strip()

            # Skip tool call progress lines
            if any(p in stripped for p in ["preparing", "┊"]):
                continue

            # Skip box-drawing borders (╭─ ... ╮ and ╰─ ... ╯)
            if stripped.startswith("╭─") or stripped.startswith("╰─"):
                continue

            # Skip lines that are purely box-drawing characters
            if stripped and all(c in "╭╮╰╯│─┃┆┊║═" for c in stripped):
                continue

            cleaned.append(line)

        result = "\n".join(cleaned).strip()

        # Deduplicate — Hermes sometimes outputs the same response twice
        # (once inside the box, once as final text)
        result = _deduplicate_response(result)

        return result if result else raw

    def _build_synthesis_prompt(self, compacted_context: str) -> str:
        """Build the prompt that feeds compacted history back to the AI."""
        instruction = self.config.resolve_synthesis_instruction()
        return (
            f"{instruction}\n\n"
            f"--- CONVERSATION HISTORY ---\n"
            f"{compacted_context}\n"
            f"--- END HISTORY ---\n\n"
            f"Continue from here. Iteration {self.state.iteration + 1} of {self.config.max_iterations}."
        )
