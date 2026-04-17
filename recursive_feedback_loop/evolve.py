"""Evolve mode -- self-refining prompt loop with optional possibility branching.

One prompt feeds itself back continuously. Each iteration:
  1. Takes the PREVIOUS OUTPUT as input
  2. Adds a refinement instruction
  3. Sends to LLM via model_choice
  4. Optionally branches via ai_possibilities if stuck or exploring
  5. The output becomes the next input

No compaction. No Hermes subprocess. Just model_choice + possibilities.

This is a different loop topology than regular RFL:
  - RFL: seed -> compact -> synthesize -> compact -> synthesize (lossy)
  - Evolve: seed -> refine -> refine -> refine (lossless, each output IS next input)
  - Evolve+branch: seed -> refine -> branch -> pick best -> refine (exploration)
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


REFINE_INSTRUCTION = """Take the following text and make it BETTER. Do NOT summarize or shorten it.
Improve it by:
1. Adding specificity (replace vague words with concrete details)
2. Adding missing considerations you can think of
3. Fixing any logical inconsistencies
4. Making the structure clearer
5. Adding actionable detail where the text is hand-wavy

OUTPUT THE FULL REVISED TEXT. Do not truncate or summarize. Do not add meta-commentary.
Just output the improved version.

---INPUT---
{input}

---END INPUT---

Output the improved version now."""

BRANCH_INSTRUCTION = """The following text is being iteratively refined. At this point, explore
ALTERNATIVE directions it could go. Generate 3-5 different variations or
extensions. For each, explain what's different and why it might be better.

After listing alternatives, pick the BEST one and output it as the refined version.

---CURRENT VERSION---
{input}

---END---

First list the alternatives, then output the best refined version."""


@dataclass
class EvolveConfig:
    seed_prompt: str
    iterations: int = 5
    branch_every: int = 0          # 0 = never branch, N = branch every N iterations
    branch_on_stagnation: bool = True  # auto-branch when output stops changing much
    stagnation_threshold: float = 0.85  # similarity ratio that counts as stagnation
    temperature: float = 0.7
    max_tokens: int = 4000
    complexity: str = "balanced"
    model: Optional[str] = None
    output_dir: str = ""
    save_snapshots: bool = True


@dataclass
class EvolveTurn:
    iteration: int
    input_text: str
    output_text: str
    branched: bool = False
    branch_alternatives: list = field(default_factory=list)
    elapsed_seconds: float = 0.0
    model_used: str = ""
    tokens_estimate: int = 0


@dataclass
class EvolveState:
    id: str = ""
    config: EvolveConfig = None
    turns: list = field(default_factory=list)
    final_output: str = ""
    converged_at: Optional[int] = None
    output_dir: str = ""

    def elapsed(self) -> float:
        return sum(t.elapsed_seconds for t in self.turns)


class EvolveRunner:
    """Run an evolve loop."""

    def __init__(self, config: EvolveConfig):
        self.config = config
        self.state = EvolveState(
            id=uuid.uuid4().hex[:8],
            config=config,
        )
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self) -> EvolveState:
        """Run the evolve loop to completion."""
        current_text = self.config.seed_prompt
        self.state.output_dir = self._resolve_output_dir()

        if self.config.save_snapshots:
            self.state.output_dir = self.state.output_dir  # ensure it exists
            Path(self.state.output_dir).mkdir(parents=True, exist_ok=True)

        for i in range(self.config.iterations):
            if self._stop:
                break

            # Decide whether to branch
            should_branch = False
            if self.config.branch_every > 0 and (i + 1) % self.config.branch_every == 0:
                should_branch = True
            if self.config.branch_on_stagnation and i > 0 and self._is_stagnant(current_text):
                should_branch = True

            turn = self._run_iteration(i, current_text, should_branch)
            self.state.turns.append(turn)

            # Save snapshot
            if self.config.save_snapshots:
                self._save_snapshot(i, turn)

            current_text = turn.output_text

            # Check convergence
            if self._is_stagnant(current_text) and i > 1:
                self.state.converged_at = i
                if not should_branch:  # already tried branching
                    break

        self.state.final_output = current_text
        self._save_final()
        return self.state

    def _run_iteration(self, iteration: int, input_text: str, branch: bool) -> EvolveTurn:
        """Run a single iteration."""
        start = time.time()

        if branch:
            prompt = BRANCH_INSTRUCTION.format(input=input_text)
        else:
            prompt = REFINE_INSTRUCTION.format(input=input_text)

        # Call model_choice
        output_text = self._call_llm(prompt)

        elapsed = time.time() - start

        # Estimate tokens (rough: 4 chars per token)
        tokens = len(output_text) // 4

        turn = EvolveTurn(
            iteration=iteration,
            input_text=input_text,
            output_text=output_text,
            branched=branch,
            elapsed_seconds=elapsed,
            tokens_estimate=tokens,
        )

        # If branching, try to extract alternatives via possibilities
        if branch:
            turn.branch_alternatives = self._extract_alternatives(output_text)

        return turn

    def _call_llm(self, prompt: str) -> str:
        """Call model_choice.generate()."""
        from model_choice import generate
        kwargs = dict(
            prompt=prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            complexity=self.config.complexity,
            use_cache=False,  # evolve shouldn't cache -- each input differs
        )
        if self.config.model:
            kwargs["model"] = self.config.model
        return generate(**kwargs)

    def _is_stagnant(self, text: str) -> bool:
        """Check if output has stopped changing significantly."""
        if len(self.state.turns) < 2:
            return False

        prev = self.state.turns[-1].output_text if self.state.turns else ""
        if not prev or not text:
            return False

        # Simple similarity: ratio of matching words
        words_a = set(prev.lower().split())
        words_b = set(text.lower().split())
        if not words_a or not words_b:
            return False

        intersection = words_a & words_b
        union = words_a | words_b
        similarity = len(intersection) / len(union) if union else 0

        return similarity >= self.config.stagnation_threshold

    def _extract_alternatives(self, text: str) -> list[dict]:
        """Try to extract alternatives from branched output.

        Uses a simple regex/heuristic approach. If the output lists numbered
        alternatives, extract them.
        """
        import re
        alternatives = []

        # Look for numbered alternatives: "1. ..." "2. ..." etc.
        pattern = re.compile(
            r'(?:^|\n)\s*(\d+)\.\s*\*{0,2}(.+?)(?:\*{0,2})\s*[-:]\s*(.+?)(?=\n\s*\d+\.|\n\n|$)',
            re.DOTALL,
        )
        for m in pattern.finditer(text):
            alternatives.append({
                "index": int(m.group(1)),
                "title": m.group(2).strip()[:100],
                "description": m.group(3).strip()[:300],
            })

        return alternatives

    def _run_possibilities(self, text: str) -> list[dict]:
        """Run ai_possibilities to explore branches of the current text.

        This calls the possibilities library directly (not the CLI) to
        explore alternative directions for the evolving prompt.
        """
        try:
            from possibilities.llm import LLMClient
            from possibilities.prompts import BRANCH_PROMPT, get_depth_guidance
        except ImportError:
            return []

        client = LLMClient(model=None, temperature=0.9, complexity="fast")

        prompt = BRANCH_PROMPT.format(
            project_context="(evolving text, no project context)",
            seed_question=f"What are alternative directions this text could evolve?\n\n{text[:2000]}",
            depth=0,
            depth_guidance=get_depth_guidance(0),
            n_min=3,
            n_max=5,
        )

        try:
            raw = client.generate(prompt)
            from possibilities.llm import parse_json_response
            branches = parse_json_response(raw)
            return branches if isinstance(branches, list) else []
        except Exception:
            return []

    def _resolve_output_dir(self) -> str:
        if self.config.output_dir:
            return self.config.output_dir
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return str(Path.cwd() / f"evolve_{timestamp}")

    def _save_snapshot(self, iteration: int, turn: EvolveTurn):
        snap_dir = Path(self.state.output_dir) / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "iteration": turn.iteration,
            "branched": turn.branched,
            "elapsed": turn.elapsed_seconds,
            "tokens_estimate": turn.tokens_estimate,
            "output_text": turn.output_text,
        }
        if turn.branch_alternatives:
            data["alternatives"] = turn.branch_alternatives
        path = snap_dir / f"iter_{iteration:03d}.json"
        path.write_text(json.dumps(data, indent=2))

    def _save_final(self):
        out = Path(self.state.output_dir)

        # conversation.md -- human-readable
        lines = [f"# Evolve Run {self.state.id}", ""]
        lines.append(f"Iterations: {len(self.state.turns)}")
        lines.append(f"Converged at: {self.state.converged_at or 'did not converge'}")
        lines.append(f"Total time: {self.state.elapsed():.1f}s")
        lines.append("")

        for turn in self.state.turns:
            lines.append(f"## Iteration {turn.iteration}")
            if turn.branched:
                lines.append(f"*(branched -- explored alternatives)*")
            lines.append(f"Time: {turn.elapsed_seconds:.1f}s | ~{turn.tokens_estimate} tokens")
            lines.append("")
            lines.append(turn.output_text)
            lines.append("")
            lines.append("---")
            lines.append("")

        (out / "conversation.md").write_text("\n".join(lines))

        # final_output.txt -- the evolved result
        (out / "final_output.txt").write_text(self.state.final_output)

        # state.json -- machine-readable
        state_data = {
            "id": self.state.id,
            "iterations": len(self.state.turns),
            "converged_at": self.state.converged_at,
            "elapsed": self.state.elapsed(),
            "turns": [
                {
                    "iteration": t.iteration,
                    "branched": t.branched,
                    "elapsed": t.elapsed_seconds,
                    "tokens": t.tokens_estimate,
                    "alternatives": t.branch_alternatives,
                    "output_preview": t.output_text[:500],
                }
                for t in self.state.turns
            ],
        }
        (out / "state.json").write_text(json.dumps(state_data, indent=2))
