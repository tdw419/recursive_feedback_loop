# RFL Autoresearch — Autonomous Tuning of the Recursive Feedback Loop

The idea: give an AI agent the RFL codebase and a fixed evaluation harness, then let it experiment autonomously. It modifies compaction strategies, prompt templates, deduper logic, and config parameters. After each change, it runs a standardized evaluation, checks if the score improved, and keeps or discards the change. You wake up to a log of experiments and (hopefully) a better RFL.

## How it works

The key insight from Karpathy's autoresearch: **fixed time budget + automated metric = autonomous research loop.**

For RFL, the challenge is the metric. Unlike LLM training (where `val_bpb` is objective), RFL produces freeform text. We solve this with a two-tier evaluation:

### Tier 1: Grounded Task (primary metric)

We have a set of **benchmark tasks** with known ground truth. Each task is a seed prompt + expected findings. The RFL runs the seed, and we score the output against ground truth.

Example task: "Audit the codebase at [path] for bugs." Ground truth: a list of known bugs with descriptions. Score = precision × recall of findings across 3 iterations.

### Tier 2: Structural Quality (secondary metric, cheap)

Measures per-iteration quality without ground truth:
- **Delta ratio**: How much new content vs repetition (target: >60% new)
- **Specificity score**: Concrete references per 1000 words (numbers, file names, function names, code blocks)
- **Depth progression**: Does each iteration go measurably deeper? (heading count, section complexity)
- **Dedup efficiency**: After deduper runs, how much was removed? (should be >10% for Hermes output)

These are fast, cheap, and correlate with quality.

## Files

- **`experiment.py`** — the evaluation harness. Runs an RFL loop with given config, scores it, outputs results. Agent modifies RFL code, then runs this.
- **`benchmarks/`** — seed prompts + ground truth for grounded evaluation
- **`program.md`** — agent instructions (the "skill" the agent follows)
- **`results.tsv`** — experiment log (not committed to git)

## Tunable parameters

These are the knobs the agent can turn:

### Compaction
- `recent_turns_verbatim` (default: 3) — how many recent turns kept full
- `medium_turns_bullets` (default: 5) — turns kept as bullets
- `summary_max_chars` (default: 500) — summary block size
- `compaction_strategy` — hierarchical | sliding_window | rolling_summary
- Token estimation heuristic (`chars/4`) — could use tiktoken

### Prompt synthesis
- `synthesis_instruction` — the "go deeper" prompt (high impact!)
- Prompt template structure (`_build_synthesis_prompt`)
- Iteration framing ("iteration N of M")

### Deduplication
- Overlap detection thresholds (200, 500, 800, 100 chunk sizes)
- Minimum content length before dedup (200 chars)
- Line-based dedup threshold (50 chars)

### Loop control
- `seed_timeout` ratio (default: 2x iteration_timeout)
- `iteration_timeout` (default: 600s)
- `max_context_tokens` (default: 8000)
- Whether to use tools on non-seed iterations

## Running

```bash
# 1. Ensure RFL is installed
cd ~/zion/projects/recursive_feedback_loop
uv sync

# 2. Run a single experiment (baseline)
uv run python -m rfl_autoresearch.experiment

# 3. Start autonomous research
# Point your agent at program.md and let it go
```
