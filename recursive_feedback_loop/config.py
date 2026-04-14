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
        "You are continuing a recursive feedback loop. You already analyzed the code in the previous iteration.\n\n"
        "YOUR RULES:\n"
        "1. DO NOT repeat or restate anything from the previous iteration. Zero repetition.\n"
        "2. You MUST go deeper. For each issue already identified, you must either:\n"
        "   - Provide the EXACT fix (complete code, line numbers, imports)\n"
        "   - Identify the ROOT CAUSE (why does this bug exist? what design assumption is wrong?)\n"
        "   - Find NEW issues that the previous iteration missed entirely\n"
        "3. Escalate specificity: if iteration 1 said 'there's a bug', iteration 2 must say "
        "'the bug is on line N in function foo() because variable X is None when Y happens'\n"
        "4. If you proposed fixes, validate them: will they break anything else? Are there edge cases?\n"
        "5. Aim for at least 3 completely NEW observations per iteration."
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
