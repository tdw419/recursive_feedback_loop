"""Configuration for the recursive feedback loop."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class LoopConfig:
    """Configuration for a recursive feedback loop run."""

    # --- Seed ---
    seed_prompt: str = ""
    seed_prompt_file: Optional[str] = None  # path to load seed from

    # --- Loop control ---
    max_iterations: int = 10
    max_runtime_seconds: int = 3600  # 1 hour default
    iteration_timeout: int = 600  # 10 min per iteration (Hermes uses tools)
    seed_timeout: Optional[int] = None  # seed iteration timeout (default: 2x iteration_timeout)

    # --- Mode ---
    mode: str = "oneshot"  # "oneshot" (fresh Hermes per iteration) or "session" (persistent tmux)
    tmux_session_name: str = "rfl_hermes"  # tmux session name for session mode

    # --- Hermes ---
    hermes_binary: str = "hermes"
    hermes_model: Optional[str] = None  # e.g. "anthropic/claude-sonnet-4"
    hermes_provider: Optional[str] = None
    hermes_profile: Optional[str] = None
    hermes_workdir: Optional[str] = None
    hermes_no_tools: bool = False  # if True, disable tools for faster text-only iterations

    # --- Compaction ---
    compaction_strategy: str = "hierarchical"  # sliding_window | rolling_summary | hierarchical
    max_context_tokens: int = 8000  # hard cap on context size sent back
    recent_turns_verbatim: int = 3  # how many recent turns to keep full
    medium_turns_bullets: int = 5  # turns to keep as bullet points
    summary_max_chars: int = 500  # max chars for the overall summary block

    # --- Prompt synthesis ---
    synthesis_instruction: str = (
        "You are in a recursive feedback loop — each iteration you see your previous thoughts. "
        "DO NOT summarize or repeat what you already said. Instead, go DEEPER. Find the next layer. "
        "Pick the most interesting thread from your previous analysis and develop it further with "
        "specific details, examples, or concrete code. If you identified problems, propose solutions. "
        "If you proposed solutions, refine them. Push toward actionable specificity."
    )
    synthesis_instruction_file: Optional[str] = None  # path to load from

    # --- Output ---
    output_dir: str = ""  # directory for logs, snapshots, exports
    save_snapshots: bool = True  # save context state after each iteration
    export_format: str = "jsonl"  # jsonl | markdown | both

    # --- Session tracking ---
    session_tag: str = "rfl"  # tag for the Hermes session source

    # --- Compaction model (for rolling_summary/hierarchical) ---
    compaction_model: Optional[str] = None  # if None, use main model
    compaction_provider: Optional[str] = None

    def resolve_seed_prompt(self) -> str:
        """Get the seed prompt from config or file."""
        if self.seed_prompt:
            return self.seed_prompt
        if self.seed_prompt_file:
            return Path(self.seed_prompt_file).read_text().strip()
        raise ValueError("No seed prompt provided. Set seed_prompt or seed_prompt_file.")

    def resolve_synthesis_instruction(self) -> str:
        """Get the synthesis instruction from config or file."""
        if self.synthesis_instruction_file:
            return Path(self.synthesis_instruction_file).read_text().strip()
        return self.synthesis_instruction

    def get_output_dir(self) -> Path:
        """Resolve output directory."""
        if self.output_dir:
            p = Path(self.output_dir)
        else:
            p = Path.cwd() / "rfl_output"
        p.mkdir(parents=True, exist_ok=True)
        return p
