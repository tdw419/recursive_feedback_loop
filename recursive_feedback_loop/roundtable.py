"""Roundtable mode -- multiple AI agents collaborate on a shared conversation.

Each "round" of the loop, every agent takes a turn:
  1. The shared conversation is compacted
  2. Each agent receives the compacted context + their role-specific prompt
  3. Each agent responds in turn (round-robin)
  4. All responses are added to the shared conversation tagged with agent name
  5. Repeat until max iterations or convergence

This is different from running N independent RFLs:
  - All agents share the SAME conversation history
  - Each agent sees what the OTHER agents found
  - They build on each other's insights, correct each other's mistakes
  - The compactor is agent-aware (preserves agent attribution)

Loop topology:
  seed -> [agent-1, agent-2, ...] -> compact -> [agent-1, agent-2, ...] -> ...

Each agent invocation is a separate Hermes subprocess. This means:
  - Agents can use different models/providers
  - A crash in one agent doesn't kill the others
  - Each agent gets a clean slate with just the compacted context
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from .agents import AgentConfig, validate_agents
from .config import LoopConfig
from .session_reader import Conversation, Turn
from .compaction import get_strategy, CompactionStrategy
from .loop_runner import LoopState, LoopRunner


# Default synthesis instruction for roundtable mode -- emphasizes cross-agent
# awareness and non-repetition.
ROUNDTABLE_SYNTHESIS = (
    "You are participating in a roundtable with multiple AI agents. "
    "You share a conversation with other agents who have different specialties.\n\n"
    "YOUR RULES:\n"
    "1. DO NOT repeat or restate anything from previous turns (yours OR other agents'). Zero repetition.\n"
    "2. Read what other agents found. Build on their insights or correct their mistakes.\n"
    "3. Focus on YOUR specialty. If other agents covered something, find what they MISSED.\n"
    "4. Escalate specificity: concrete line numbers, exact function names, precise root causes.\n"
    "5. If you agree with another agent's finding, go DEEPER into it -- don't just say 'I agree'.\n"
    "6. Aim for at least 2 completely NEW observations that no previous turn has made.\n"
)


@dataclass
class RoundTableConfig:
    """Configuration for a roundtable run."""
    agents: List[AgentConfig]
    seed_prompt: str = ""
    seed_prompt_file: Optional[str] = None
    max_rounds: int = 5           # outer iterations (each round = N agent turns)
    max_runtime_seconds: int = 7200  # 2 hours default (more agents = more time)
    iteration_timeout: int = 600  # per-agent timeout
    seed_timeout: Optional[int] = None

    # Hermes settings (defaults for agents that don't override)
    hermes_binary: str = "hermes"
    hermes_workdir: Optional[str] = None
    hermes_no_tools: bool = False

    # Compaction
    compaction_strategy: str = "hierarchical"
    max_context_tokens: int = 10000  # slightly larger default for multi-agent
    recent_turns_verbatim: int = 4   # keep more recent turns (one per agent)
    medium_turns_bullets: int = 6
    summary_max_chars: int = 600

    # Prompt synthesis
    synthesis_instruction: str = ROUNDTABLE_SYNTHESIS
    synthesis_instruction_file: Optional[str] = None

    # Output
    output_dir: str = ""
    save_snapshots: bool = True
    export_format: str = "both"

    # Concurrency
    run_id: str = ""

    # Compaction model
    compaction_model: Optional[str] = None
    compaction_provider: Optional[str] = None

    def resolve_seed_prompt(self) -> str:
        if self.seed_prompt:
            return self.seed_prompt
        if self.seed_prompt_file:
            return Path(self.seed_prompt_file).read_text().strip()
        raise ValueError("No seed prompt provided.")

    def resolve_synthesis_instruction(self) -> str:
        if self.synthesis_instruction_file:
            return Path(self.synthesis_instruction_file).read_text().strip()
        return self.synthesis_instruction

    def get_output_dir(self) -> Path:
        if self.output_dir:
            p = Path(self.output_dir)
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            p = Path.cwd() / f"rfl_roundtable_{timestamp}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_run_id(self) -> str:
        if self.run_id:
            return self.run_id
        return f"roundtable_{os.getpid()}_{int(time.time())}"

    def get_lockfile_path(self) -> Path:
        return self.get_output_dir() / ".rfl.lock"

    def get_lockfile_data(self) -> dict:
        return {
            "pid": os.getpid(),
            "run_id": self.get_run_id(),
            "mode": "roundtable",
            "agents": [a.name for a in self.agents],
            "workdir": str(Path(self.hermes_workdir).resolve()) if self.hermes_workdir else "",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    def _seed_timeout(self) -> int:
        if self.seed_timeout:
            return self.seed_timeout
        return self.iteration_timeout * 2


class RoundTableRunner:
    """Runs a multi-agent roundtable loop."""

    def __init__(self, config: RoundTableConfig):
        self.config = config
        self.state: Optional[LoopState] = None
        self.compactor: Optional[CompactionStrategy] = None
        self._running = False

    def run(self) -> LoopState:
        """Execute the full roundtable loop."""
        self._running = True
        output_dir = self.config.get_output_dir()
        self.state = LoopState(output_dir)

        # Acquire PID lock
        if not self._acquire_lock():
            self._running = False
            return self.state

        # Initialize compactor
        compactor_kwargs = {
            "recent_turns": self.config.recent_turns_verbatim,
            "summary_max_chars": self.config.summary_max_chars,
            "compaction_model": self.config.compaction_model,
            "compaction_provider": self.config.compaction_provider,
        }
        if self.config.compaction_strategy == "hierarchical":
            compactor_kwargs["medium_turns"] = self.config.medium_turns_bullets
        elif self.config.compaction_strategy == "sliding_window":
            compactor_kwargs["keep_turns"] = self.config.recent_turns_verbatim * 2

        self.compactor = get_strategy(self.config.compaction_strategy, **compactor_kwargs)

        self.state.log("roundtable_start", {
            "config": {
                "agents": [a.display() for a in self.config.agents],
                "max_rounds": self.config.max_rounds,
                "compaction": self.config.compaction_strategy,
                "token_budget": self.config.max_context_tokens,
                "run_id": self.config.get_run_id(),
                "output_dir": str(output_dir),
            }
        })

        try:
            return self._run_loop()
        finally:
            self._release_lock()

    def _run_loop(self) -> LoopState:
        """Main loop: seed -> rounds of agent turns -> compact -> repeat."""

        # --- Seed round: every agent responds to the seed prompt ---
        seed = self.config.resolve_seed_prompt()
        self.state.log("seed_prompt", {"content_preview": seed[:200]})
        self.state.conversation.add("user", seed, iteration=0)

        instruction = self.config.resolve_synthesis_instruction()

        for idx, agent in enumerate(self.config.agents):
            if not self._running:
                break

            agent_prompt = self._build_agent_prompt(agent, seed, instruction, is_seed=True)
            self.state.log("agent_seed_start", {"agent": agent.name, "agent_idx": idx})

            response = self._run_agent_query(agent, agent_prompt, is_seed=True)
            if response:
                self.state.conversation.add("assistant", response, iteration=0, agent=agent.name)
                self.state.log("agent_seed_response", {
                    "agent": agent.name,
                    "tokens": len(response) // 4,
                    "content_preview": response[:200],
                })
            else:
                self.state.log("agent_seed_empty", {"agent": agent.name})

        # --- Rounds 1..N ---
        for round_num in range(1, self.config.max_rounds):
            if not self._running:
                self.state.log("roundtable_stopped", {"reason": "external_stop"})
                break

            if self.state.elapsed() > self.config.max_runtime_seconds:
                self.state.log("roundtable_stopped", {"reason": "timeout"})
                break

            self.state.iteration = round_num

            # Compact the shared conversation
            compacted = self.compactor.compact(self.state.conversation, self.config.max_context_tokens)
            self.state.log("compacted", {
                "strategy": self.compactor.name(),
                "compacted_tokens": len(compacted) // 4,
                "original_tokens": self.state.conversation.total_tokens_estimate(),
            })

            # Save snapshot before this round
            if self.config.save_snapshots:
                snap_path = self.state.save_snapshot()
                self.state.log("snapshot_saved", {"path": str(snap_path)})

            # Each agent takes a turn
            for idx, agent in enumerate(self.config.agents):
                if not self._running:
                    break

                agent_prompt = self._build_agent_prompt(
                    agent, compacted, instruction,
                    is_seed=False, round_num=round_num,
                )

                self.state.log("agent_turn_start", {
                    "agent": agent.name, "round": round_num,
                })

                response = self._run_agent_query(agent, agent_prompt, is_seed=False)
                if response:
                    self.state.conversation.add("assistant", response, iteration=round_num, agent=agent.name)
                    self.state.log("agent_turn_response", {
                        "agent": agent.name,
                        "round": round_num,
                        "tokens": len(response) // 4,
                        "content_preview": response[:200],
                    })
                else:
                    self.state.log("agent_turn_empty", {
                        "agent": agent.name, "round": round_num,
                    })

        return self._finalize()

    def _build_agent_prompt(
        self,
        agent: AgentConfig,
        context: str,
        instruction: str,
        is_seed: bool = False,
        round_num: int = 0,
    ) -> str:
        """Build the prompt for a specific agent.

        Includes:
        - The agent's role/persona
        - The shared context (seed or compacted history)
        - The synthesis instruction
        - Round/iteration metadata
        """
        parts = []

        # Agent identity
        if agent.role:
            parts.append(f"YOUR ROLE: {agent.role}")
            parts.append(f"YOUR NAME: {agent.name}")
            parts.append("")

        parts.append(instruction)
        parts.append("")

        if is_seed:
            parts.append("--- TASK ---")
            parts.append(context)
            parts.append("--- END TASK ---")
        else:
            parts.append("--- SHARED CONVERSATION HISTORY ---")
            parts.append(context)
            parts.append("--- END HISTORY ---")

        total_agents = len(self.config.agents)
        parts.append("")
        if is_seed:
            parts.append(
                f"You are agent '{agent.name}' of {total_agents}. "
                f"This is the INITIAL TASK. Respond with your analysis."
            )
        else:
            parts.append(
                f"This is round {round_num} of {self.config.max_rounds}. "
                f"You are agent '{agent.name}' of {total_agents}. "
                f"Do NOT repeat what you or others already said. Go deeper, find new angles, be specific."
            )

        return "\n".join(parts)

    def _run_agent_query(self, agent: AgentConfig, prompt: str, is_seed: bool = False) -> Optional[str]:
        """Run a single Hermes query for a specific agent."""
        model = agent.model
        provider = agent.provider
        profile = agent.profile

        cmd = [self.config.hermes_binary, "chat", "-q", prompt, "-Q", "-t", ""]
        if model:
            cmd.extend(["-m", model])
        elif self.config.hermes_binary == "hermes":
            # No agent-specific model, no fallback either -- use Hermes default
            pass

        if provider:
            cmd.extend(["--provider", provider])

        if profile:
            cmd.extend(["-p", profile])

        timeout = self.config._seed_timeout() if is_seed else self.config.iteration_timeout

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.config.hermes_workdir,
            )
            if result.returncode == 0:
                raw = result.stdout.strip()
                if raw:
                    return LoopRunner._clean_hermes_output(raw)
                return None
            else:
                if self.state:
                    self.state.log("agent_hermes_error", {
                        "agent": agent.name,
                        "returncode": result.returncode,
                        "stderr": result.stderr[:500] if result.stderr else None,
                    })
                return None
        except subprocess.TimeoutExpired:
            if self.state:
                self.state.log("agent_hermes_timeout", {
                    "agent": agent.name,
                    "timeout": timeout,
                })
            return None
        except Exception as e:
            if self.state:
                self.state.log("agent_hermes_exception", {
                    "agent": agent.name,
                    "error": str(e),
                })
            return None

    def stop(self):
        """Gracefully stop the roundtable."""
        self._running = False

    def _finalize(self) -> LoopState:
        """Export results and finalize."""
        self.state.log("roundtable_end", {
            "total_rounds": self.state.iteration + 1,
            "total_tokens": self.state.conversation.total_tokens_estimate(),
            "elapsed_seconds": round(self.state.elapsed(), 1),
            "agents": self.state.conversation.agent_names(),
        })

        if self.config.save_snapshots:
            self.state.save_snapshot()

        if self.config.export_format in ("markdown", "both"):
            md_path = self._export_markdown()
            self.state.log("exported_markdown", {"path": str(md_path)})
        if self.config.export_format in ("jsonl", "both"):
            jsonl_path = self.state.export_jsonl()
            self.state.log("exported_jsonl", {"path": str(jsonl_path)})

        self._running = False
        return self.state

    def _export_markdown(self) -> Path:
        """Export full conversation as markdown with agent attribution."""
        path = self.state.output_dir / "conversation.md"
        lines = [
            "# Roundtable RFL -- Conversation Export",
            "",
            f"Generated: {datetime.now().isoformat()}",
            f"Agents: {', '.join(a.display() for a in self.config.agents)}",
            f"Rounds: {self.state.iteration + 1}",
            "",
        ]

        for turn in self.state.conversation.turns:
            role = turn.role.upper()
            agent_tag = f" [{turn.agent}]" if turn.agent else ""
            lines.append(f"## {role}{agent_tag} (Round {turn.iteration})")
            lines.append("")
            lines.append(turn.content)
            lines.append("")
            lines.append("---")
            lines.append("")

        path.write_text("\n".join(lines))
        return path

    def _acquire_lock(self) -> bool:
        """Acquire a PID lockfile."""
        lockpath = self.config.get_lockfile_path()

        if lockpath.exists():
            try:
                data = json.loads(lockpath.read_text())
                existing_pid = data.get("pid", 0)
                if existing_pid:
                    try:
                        os.kill(existing_pid, 0)
                        print(
                            f"ERROR: Output dir {self.config.get_output_dir()} is locked by "
                            f"PID {existing_pid} (run {data.get('run_id', '?')})",
                            file=sys.stderr,
                        )
                        return False
                    except (ProcessLookupError, PermissionError):
                        lockpath.unlink(missing_ok=True)
            except (json.JSONDecodeError, OSError):
                lockpath.unlink(missing_ok=True)

        try:
            lockpath.write_text(json.dumps(self.config.get_lockfile_data(), indent=2))
            return True
        except OSError as e:
            print(f"WARNING: Could not write lockfile: {e}", file=sys.stderr)
            return True

    def _release_lock(self):
        """Release the PID lockfile."""
        lockpath = self.config.get_lockfile_path()
        try:
            if lockpath.exists():
                data = json.loads(lockpath.read_text())
                if data.get("pid") == os.getpid():
                    lockpath.unlink()
        except (json.JSONDecodeError, OSError):
            pass
