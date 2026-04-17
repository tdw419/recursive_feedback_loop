"""Build mode -- AI builds a project by feeding its own progress back.

Give it a project idea. The AI designs, builds, tests, and iterates.
Each iteration the AI sees what it produced last time, what files exist
in the project, and continues building however it wants.

When the AI wants to explore alternatives, it can trigger possibilities
by including [EXPLORE: question] in its output. The loop detects this,
runs ai_possibilities, and feeds the branches back as context.

Architecture:
  Iteration 0: seed prompt -> Hermes (tools) -> design + start building
  Iteration N: project state + last output + explore results -> Hermes -> continue
  Between iterations: detect [EXPLORE] triggers, run possibilities if found

Different from RFL:
  - No compaction. Each iteration gets: seed + project state + last output.
  - No prescriptive synthesis. The AI decides what to do next.
  - Possibilities integration for creative branching.

Different from evolve:
  - Uses Hermes subprocess (has tools -- reads/writes files).
  - Reads actual project state between iterations.
  - The AI builds a real project, not just refined text.
"""

import json
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


CONTINUE_TEMPLATE = """You are autonomously building a project. Here's your context:

ORIGINAL GOAL:
{seed}

PROJECT STATE:
{project_state}

YOUR PREVIOUS OUTPUT (iteration {prev_iteration}):
{prev_output}
{explore_context}

Continue building. You have full creative control. Read files, write files,
run commands. Decide what to do next based on what you've built so far.
If you're stuck or want to explore alternatives, include this line in your output:
[EXPLORE: your question here]
The system will explore alternatives and feed them back to you.
If you think the project is complete, say [DONE] on its own line."""

EXPLORE_INJECT_TEMPLATE = """

EXPLORATION RESULTS (alternatives the system found):
{explore_results}
Consider these alternatives and pick the best path forward, or ignore them
if you have a better idea. You're in control."""


@dataclass
class BuildConfig:
    seed_prompt: str
    workdir: str = "."               # where Hermes runs (project dir)
    iterations: int = 8
    hermes_binary: str = "hermes"
    hermes_model: Optional[str] = None
    hermes_provider: Optional[str] = None
    iteration_timeout: int = 900     # seconds per iteration
    explore_enabled: bool = True     # enable [EXPLORE] triggers
    explore_depth: int = 1           # possibilities exploration depth
    explore_max_branches: int = 5
    output_dir: str = ""
    save_snapshots: bool = True


@dataclass
class BuildTurn:
    iteration: int
    prompt: str
    output: str
    explored: bool = False
    explore_question: str = ""
    explore_branches: list = field(default_factory=list)
    elapsed_seconds: float = 0.0
    hermes_exit_code: int = 0


@dataclass
class BuildState:
    id: str = ""
    config: BuildConfig = None
    turns: list = field(default_factory=list)
    final_output: str = ""
    project_path: str = ""
    output_dir: str = ""

    def elapsed(self) -> float:
        return sum(t.elapsed_seconds for t in self.turns)


class BuildRunner:
    """Run an autonomous build loop."""

    def __init__(self, config: BuildConfig):
        self.config = config
        self.state = BuildState(
            id=uuid.uuid4().hex[:8],
            config=config,
            project_path=str(Path(config.workdir).resolve()),
        )
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self) -> BuildState:
        """Run the build loop."""
        self.state.output_dir = self._resolve_output_dir()
        Path(self.state.output_dir).mkdir(parents=True, exist_ok=True)

        prev_output = ""
        explore_context = ""

        for i in range(self.config.iterations):
            if self._stop:
                break

            # Build the prompt for this iteration
            if i == 0:
                prompt = self.config.seed_prompt
            else:
                project_state = self._gather_project_state()
                prompt = CONTINUE_TEMPLATE.format(
                    seed=self.config.seed_prompt[:2000],
                    project_state=project_state,
                    prev_iteration=i - 1,
                    prev_output=prev_output[:6000],
                    explore_context=explore_context,
                )

            # Run Hermes
            turn = self._run_hermes(i, prompt)
            self.state.turns.append(turn)
            prev_output = turn.output

            # Save snapshot
            if self.config.save_snapshots:
                self._save_snapshot(i, turn)

            # Check for [DONE]
            if "[DONE]" in turn.output:
                print(f"  AI declared project done at iteration {i}")
                break

            # Check for [EXPLORE] triggers
            explore_context = ""
            if self.config.explore_enabled:
                question = self._detect_explore_trigger(turn.output)
                if question:
                    print(f"  AI wants to explore: {question[:80]}")
                    branches = self._run_possibilities(question)
                    if branches:
                        branch_text = "\n".join(
                            f"  - {b.get('title', '?')}: {b.get('description', '')[:120]}"
                            for b in branches
                        )
                        explore_context = EXPLORE_INJECT_TEMPLATE.format(
                            explore_results=branch_text
                        )
                        turn.explored = True
                        turn.explore_question = question
                        turn.explore_branches = branches

            # Print progress
            print(f"  Iter {i}: {len(turn.output)} chars, {turn.elapsed_seconds:.0f}s"
                  f"{' [EXPLORED]' if turn.explored else ''}"
                  f"{' [DONE]' if '[DONE]' in turn.output else ''}")

        self.state.final_output = prev_output
        self._save_final()
        return self.state

    def _run_hermes(self, iteration: int, prompt: str) -> BuildTurn:
        """Spawn Hermes subprocess with the prompt."""
        start = time.time()

        cmd = [self.config.hermes_binary, "-q", "-Q", "-t", ""]
        if self.config.hermes_model:
            cmd.extend(["-m", self.config.hermes_model])
        if self.config.hermes_provider:
            cmd.extend(["--provider", self.config.hermes_provider])

        # Write prompt to temp file to avoid ARG_MAX issues
        prompt_file = Path(self.state.output_dir) / f"prompt_{iteration}.txt"
        prompt_file.write_text(prompt)

        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.config.iteration_timeout,
                cwd=self.config.workdir,
            )
            output = result.stdout or ""
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            output = ""
            exit_code = -1
        except Exception as e:
            output = f"Error running Hermes: {e}"
            exit_code = -1

        # Clean Hermes output (deduplicate boxed + final)
        output = self._clean_output(output)

        elapsed = time.time() - start
        return BuildTurn(
            iteration=iteration,
            prompt=prompt[:500],  # save first 500 chars of prompt
            output=output,
            elapsed_seconds=elapsed,
            hermes_exit_code=exit_code,
        )

    def _clean_output(self, text: str) -> str:
        """Clean Hermes output -- deduplicate, strip artifacts."""
        if not text:
            return ""

        # Remove ANSI escape codes
        text = re.sub(r'\x1b\[[0-9;]*m', '', text)

        # Hermes dedup: if second half matches first half
        mid = len(text) // 2
        if mid > 100:
            first = text[:mid].strip()
            second = text[mid:].strip()
            overlap = min(200, len(first), len(second))
            if overlap > 50:
                if first[:overlap].replace(" ", "") == second[:overlap].replace(" ", ""):
                    return first

        return text.strip()

    def _gather_project_state(self) -> str:
        """Gather current project state for context injection."""
        lines = []
        proj = Path(self.config.workdir)

        # File listing
        files = []
        for root, dirs, filenames in os.walk(proj):
            dirs[:] = [d for d in dirs
                       if d not in {'.git', '__pycache__', 'node_modules', '.venv',
                                    'venv', 'dist', 'build', 'target', '.mypy_cache'}
                       and not d.startswith('rfl_') and not d.startswith('evolve_')]
            for fn in filenames:
                fp = Path(root) / fn
                rel = fp.relative_to(proj)
                files.append((str(rel), fp.stat().st_size))

        if files:
            files.sort(key=lambda x: x[0])
            lines.append(f"Files ({len(files)} total):")
            for path, size in files[:40]:
                lines.append(f"  {path} ({size} bytes)")
            if len(files) > 40:
                lines.append(f"  ... and {len(files) - 40} more")

        # Git log (recent commits)
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "-5"],
                capture_output=True, text=True, timeout=5,
                cwd=self.config.workdir,
            )
            if result.returncode == 0 and result.stdout.strip():
                lines.append("\nRecent commits:")
                for line in result.stdout.strip().splitlines()[:5]:
                    lines.append(f"  {line}")
        except Exception:
            pass

        # Git diff (uncommitted changes)
        try:
            result = subprocess.run(
                ["git", "diff", "--stat"],
                capture_output=True, text=True, timeout=5,
                cwd=self.config.workdir,
            )
            if result.returncode == 0 and result.stdout.strip():
                lines.append("\nUncommitted changes:")
                lines.append(result.stdout.strip()[:500])
        except Exception:
            pass

        return "\n".join(lines) if lines else "(empty project)"

    def _detect_explore_trigger(self, output: str) -> str:
        """Check if the AI output contains an [EXPLORE: question] trigger."""
        m = re.search(r'\[EXPLORE:\s*(.+?)\]', output)
        if m:
            return m.group(1).strip()
        return ""

    def _run_possibilities(self, question: str) -> list[dict]:
        """Run ai_possibilities to explore the AI's question."""
        try:
            from possibilities.llm import LLMClient
            from possibilities.prompts import BRANCH_PROMPT, get_depth_guidance
            from possibilities.llm import parse_json_response
        except ImportError:
            print("  [possibilities not available, skipping explore]")
            return []

        try:
            client = LLMClient(model=None, temperature=0.9, complexity="fast")
            prompt = BRANCH_PROMPT.format(
                project_context=f"Project at {self.config.workdir}. AI is autonomously building it.",
                seed_question=question,
                depth=0,
                depth_guidance=get_depth_guidance(0),
                n_min=3,
                n_max=self.config.explore_max_branches,
            )
            raw = client.generate(prompt)
            branches = parse_json_response(raw)
            return branches if isinstance(branches, list) else []
        except Exception as e:
            print(f"  [explore failed: {e}]")
            return []

    def _resolve_output_dir(self) -> str:
        if self.config.output_dir:
            return self.config.output_dir
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return str(Path.cwd() / f"build_{timestamp}")

    def _save_snapshot(self, iteration: int, turn: BuildTurn):
        snap_dir = Path(self.state.output_dir) / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "iteration": turn.iteration,
            "explored": turn.explored,
            "explore_question": turn.explore_question,
            "elapsed": turn.elapsed_seconds,
            "hermes_exit_code": turn.hermes_exit_code,
            "output": turn.output,
        }
        if turn.explore_branches:
            data["explore_branches"] = turn.explore_branches
        path = snap_dir / f"iter_{iteration:03d}.json"
        path.write_text(json.dumps(data, indent=2))

    def _save_final(self):
        out = Path(self.state.output_dir)

        # conversation.md
        lines = [f"# Build Run {self.state.id}", ""]
        lines.append(f"Seed: {self.config.seed_prompt[:200]}")
        lines.append(f"Workdir: {self.config.workdir}")
        lines.append(f"Iterations: {len(self.state.turns)}")
        lines.append(f"Time: {self.state.elapsed():.1f}s")
        explored = sum(1 for t in self.state.turns if t.explored)
        if explored:
            lines.append(f"Explored: {explored} iterations")
        lines.append("")

        for turn in self.state.turns:
            lines.append(f"## Iteration {turn.iteration}")
            if turn.explored:
                lines.append(f"*Explored: {turn.explore_question}*")
            lines.append(f"Time: {turn.elapsed_seconds:.1f}s | Exit: {turn.hermes_exit_code}")
            lines.append("")
            lines.append(turn.output)
            lines.append("")
            lines.append("---")
            lines.append("")

        (out / "conversation.md").write_text("\n".join(lines))

        # build_log.jsonl
        with open(out / "build_log.jsonl", "w") as f:
            for turn in self.state.turns:
                entry = {
                    "iteration": turn.iteration,
                    "elapsed": turn.elapsed_seconds,
                    "output_len": len(turn.output),
                    "explored": turn.explored,
                    "explore_question": turn.explore_question,
                    "hermes_exit_code": turn.hermes_exit_code,
                }
                f.write(json.dumps(entry) + "\n")
