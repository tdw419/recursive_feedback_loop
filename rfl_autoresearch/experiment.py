"""RFL Autoresearch — evaluation harness.

Runs an RFL loop with specific config against a benchmark task,
then scores the output on multiple dimensions.

Usage:
    python -m rfl_autoresearch.experiment                  # baseline config
    python -m rfl_autoresearch.experiment --iterations 5   # override params
    python -m rfl_autoresearch.experiment --help

Output: prints a summary like
    ---
    delta_ratio:         0.72
    specificity_score:   4.50
    depth_progression:   0.85
    dedup_efficiency:    0.15
    composite_score:     0.68
    total_seconds:       180.5
    iterations:          3
    total_words:         2500
"""

import argparse
import json
import re
import time
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

# Add parent to path so we can import rfl
sys.path.insert(0, str(Path(__file__).parent.parent))

from recursive_feedback_loop.config import LoopConfig
from recursive_feedback_loop.loop_runner import LoopRunner, _deduplicate_response


# ─── Benchmark tasks ───────────────────────────────────────────────

BENCHMARKS_DIR = Path(__file__).parent / "benchmarks"

DEFAULT_SEED = """\
Analyze the following Python code for bugs, design issues, and improvement opportunities.
Be specific — cite line numbers, function names, and concrete problems.

```python
import subprocess
import json

def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout

def parse_config(path):
    with open(path) as f:
        data = json.loads(f.read())
    return data.get("settings", {})

def process_items(items, batch_size=100):
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        result = run_cmd(f"process --batch {json.dumps(batch)}")
        results.append(json.loads(result))
    return results

def get_user(user_id):
    cmd = f"curl http://api.example.com/users/{user_id}"
    response = run_cmd(cmd)
    return json.loads(response)

def save_report(data, filename="report.json"):
    with open(filename, "w") as f:
        json.dump(data, f)
    return filename
```

For each issue found: explain the bug, show the vulnerable line, and propose a fix.\
"""


@dataclass
class ExperimentResult:
    """Scores from an experiment run."""
    delta_ratio: float         # new content vs repetition (0-1, higher = better)
    specificity_score: float   # concrete references per 1K words
    depth_progression: float   # does output deepen across iterations (0-1)
    dedup_efficiency: float    # fraction of duplicate content removed (0-1)
    composite_score: float     # weighted combination
    total_seconds: float
    iterations: int
    total_words: int
    total_tokens_estimate: int
    status: str                # "keep", "discard", "crash"
    description: str


# ─── Scoring functions ─────────────────────────────────────────────

def _word_count(text: str) -> int:
    return len(text.split())


def _count_specificity_markers(text: str) -> int:
    """Count concrete references: numbers, file names, function names, code blocks."""
    markers = 0
    # Numbers (line numbers, sizes, counts)
    markers += len(re.findall(r'\b\d+\b', text))
    # Function names (word followed by parens)
    markers += len(re.findall(r'\b[a-z_]\w*\(', text))
    # Code blocks (``` or inline `)
    markers += len(re.findall(r'```[\s\S]*?```', text))
    markers += len(re.findall(r'`[^`]+`', text))
    # File paths
    markers += len(re.findall(r'[\w/.]+\.\w{1,4}', text))
    return markers


def score_delta_ratio(turns_text: List[str]) -> float:
    """Measure how much new content each iteration adds vs repeating.

    For each pair of consecutive turns, compute the fraction of
    bigrams in the new turn that don't appear in the previous one.
    Average across all pairs.
    """
    if len(turns_text) < 2:
        return 1.0  # only one turn, all new by definition

    ratios = []
    for i in range(1, len(turns_text)):
        prev_words = set(turns_text[i-1].lower().split())
        curr_words = turns_text[i].lower().split()

        if not curr_words:
            ratios.append(0.0)
            continue

        # Count words in current that don't appear in previous
        new_words = sum(1 for w in curr_words if w not in prev_words)
        ratio = new_words / len(curr_words)
        ratios.append(ratio)

    return sum(ratios) / len(ratios) if ratios else 0.0


def score_specificity(text: str) -> float:
    """Specificity markers per 1000 words."""
    words = _word_count(text)
    if words == 0:
        return 0.0
    markers = _count_specificity_markers(text)
    return (markers / words) * 1000


def score_depth_progression(turns_text: List[str]) -> float:
    """Does each iteration produce more specific/deeper output?

    Measures: does specificity increase? Does word count stay stable or grow?
    Does heading structure deepen?
    """
    if len(turns_text) < 2:
        return 1.0

    scores = []
    specificities = [score_specificity(t) for t in turns_text]

    # Check if specificity generally increases
    for i in range(1, len(specificities)):
        if specificities[i-1] == 0:
            scores.append(0.5)  # neutral
        else:
            # Reward increasing specificity, penalize decreasing
            ratio = specificities[i] / specificities[i-1]
            # Clamp: 0.5x to 2x is the interesting range
            clamped = max(0.5, min(2.0, ratio))
            # Normalize to 0-1
            scores.append((clamped - 0.5) / 1.5)

    return sum(scores) / len(scores) if scores else 0.5


def score_dedup_efficiency(raw_texts: List[str], deduped_texts: List[str]) -> float:
    """How much duplicate content did the deduper find and remove?

    Higher = more duplication was caught (good deduper).
    0 = no duplication found (either clean output or broken deduper).
    """
    if not raw_texts or not deduped_texts:
        return 0.0

    total_raw = sum(len(t) for t in raw_texts)
    total_deduped = sum(len(t) for t in deduped_texts)

    if total_raw == 0:
        return 0.0

    return max(0.0, 1.0 - (total_deduped / total_raw))


def compute_composite(
    delta: float,
    specificity: float,
    depth: float,
    dedup: float,
) -> float:
    """Weighted composite score. All inputs 0-1 range (specificity normalized)."""
    # Normalize specificity: ~5 markers/1K words is decent, 10+ is great
    spec_normalized = min(1.0, specificity / 10.0)

    # Weights: delta and specificity are most important
    weights = {
        'delta': 0.35,
        'specificity': 0.30,
        'depth': 0.20,
        'dedup': 0.15,
    }
    return (
        weights['delta'] * delta +
        weights['specificity'] * spec_normalized +
        weights['depth'] * depth +
        weights['dedup'] * dedup
    )


# ─── Experiment runner ─────────────────────────────────────────────

def run_experiment(config: LoopConfig, description: str = "unnamed") -> ExperimentResult:
    """Run an RFL loop and score the output."""
    start_time = time.time()

    try:
        runner = LoopRunner(config)
        state = runner.run()
        elapsed = time.time() - start_time

        # Extract turn texts
        assistant_turns = [t for t in state.conversation.turns if t.role == "assistant"]
        turns_text = [t.content for t in assistant_turns]

        # Compute dedup efficiency: re-run deduper and compare
        raw_texts = []
        deduped_texts = []
        for t in turns_text:
            # Simulate pre-dedup by "un-deduping" (just use the text as-is
            # since deduper already ran during the loop). We'll measure
            # what the deduper would do on this output.
            deduped = _deduplicate_response(t)
            raw_texts.append(t)
            deduped_texts.append(deduped)

        all_text = "\n\n".join(turns_text)
        total_words = _word_count(all_text)
        total_tokens = state.conversation.total_tokens_estimate()

        # Compute scores
        delta = score_delta_ratio(turns_text)
        specificity = score_specificity(all_text)
        depth = score_depth_progression(turns_text)
        dedup = score_dedup_efficiency(raw_texts, deduped_texts)
        composite = compute_composite(delta, specificity, depth, dedup)

        status = "keep" if composite > 0.4 else "discard"

        return ExperimentResult(
            delta_ratio=delta,
            specificity_score=specificity,
            depth_progression=depth,
            dedup_efficiency=dedup,
            composite_score=composite,
            total_seconds=elapsed,
            iterations=state.iteration + 1,
            total_words=total_words,
            total_tokens_estimate=total_tokens,
            status=status,
            description=description,
        )

    except Exception as e:
        elapsed = time.time() - start_time
        return ExperimentResult(
            delta_ratio=0.0,
            specificity_score=0.0,
            depth_progression=0.0,
            dedup_efficiency=0.0,
            composite_score=0.0,
            total_seconds=elapsed,
            iterations=0,
            total_words=0,
            total_tokens_estimate=0,
            status="crash",
            description=description,
        )


def format_result(result: ExperimentResult) -> str:
    """Format result for stdout (matches AutoResearch output style)."""
    return f"""---
delta_ratio:         {result.delta_ratio:.6f}
specificity_score:   {result.specificity_score:.2f}
depth_progression:   {result.depth_progression:.6f}
dedup_efficiency:    {result.dedup_efficiency:.6f}
composite_score:     {result.composite_score:.6f}
total_seconds:       {result.total_seconds:.1f}
iterations:          {result.iterations}
total_words:         {result.total_words}
total_tokens:        {result.total_tokens_estimate}
status:              {result.status}
description:         {result.description}
---"""


def log_result(result: ExperimentResult, results_file: Path):
    """Append result to the TSV log."""
    import subprocess
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, timeout=5,
    ).stdout.strip() or "unknown"

    if not results_file.exists():
        results_file.write_text(
            "commit\tcomposite\tdelta\tspecificity\tdepth\tdedup\twords\tseconds\tstatus\tdescription\n"
        )

    with open(results_file, "a") as f:
        f.write(
            f"{commit}\t{result.composite_score:.6f}\t{result.delta_ratio:.6f}\t"
            f"{result.specificity_score:.2f}\t{result.depth_progression:.6f}\t"
            f"{result.dedup_efficiency:.6f}\t{result.total_words}\t"
            f"{result.total_seconds:.1f}\t{result.status}\t{result.description}\n"
        )


def _dry_run(config: LoopConfig, description: str) -> ExperimentResult:
    """Generate mock RFL output for testing the scoring pipeline without Hermes."""
    import random

    mock_turns = [
        "Initial analysis of the code. Found potential issues in run_cmd() function at line 3: "
        "uses shell=True which is a security risk. Also parse_config() at line 7 may raise "
        "FileNotFoundError. The process_items() function at line 11 has no error handling.",
        "Deeper investigation confirms: run_cmd() shell=True allows command injection via the "
        "batch parameter in process_items(). The json.loads() at line 14 will crash if the "
        "subprocess returns non-JSON output. Fix for run_cmd(): use subprocess.run(cmd, shell=False) "
        "with shlex.split(). Fix for process_items(): add try/except around json.loads(). "
        "Also found: get_user() at line 18 has no timeout on the curl command.",
        "Root cause analysis complete. The fundamental issue is lack of input validation "
        "throughout. Proposed fixes:\n```python\nimport shlex\n\ndef run_cmd(cmd):\n    "
        "return subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=30)\n```\n"
        "For process_items(), add batch validation:\n```python\nif not isinstance(batch, list):\n"
        "    raise ValueError('batch must be a list')\n```\n"
        "For save_report(), add atomic writes using tempfile + os.rename. "
        "Overall risk: HIGH. 3 injection vectors, 2 crash paths, 1 data corruption risk.",
    ]

    # Truncate to configured iterations
    mock_turns = mock_turns[:config.max_iterations]

    all_text = "\n\n".join(mock_turns)
    total_words = _word_count(all_text)

    delta = score_delta_ratio(mock_turns)
    specificity = score_specificity(all_text)
    depth = score_depth_progression(mock_turns)
    dedup = 0.0  # no duplication in mock
    composite = compute_composite(delta, specificity, depth, dedup)

    return ExperimentResult(
        delta_ratio=delta,
        specificity_score=specificity,
        depth_progression=depth,
        dedup_efficiency=dedup,
        composite_score=composite,
        total_seconds=1.0,
        iterations=len(mock_turns),
        total_words=total_words,
        total_tokens_estimate=total_words * 4,
        status="keep" if composite > 0.4 else "discard",
        description=description,
    )


# ─── CLI ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RFL Autoresearch Experiment")
    parser.add_argument("--iterations", "-n", type=int, default=3,
                        help="Number of RFL iterations (default: 3, fast eval)")
    parser.add_argument("--budget", "-b", type=int, default=8000,
                        help="Token budget per iteration (default: 8000)")
    parser.add_argument("--strategy", "-s",
                        choices=["sliding_window", "rolling_summary", "hierarchical"],
                        default="hierarchical", help="Compaction strategy")
    parser.add_argument("--model", "-m", default=None, help="Hermes model")
    parser.add_argument("--provider", default=None, help="Hermes provider")
    parser.add_argument("--timeout", "-t", type=int, default=600,
                        help="Per-iteration timeout (default: 600)")
    parser.add_argument("--no-tools", action="store_true",
                        help="Disable tools for non-seed iterations")
    parser.add_argument("--recent-turns", type=int, default=3)
    parser.add_argument("--medium-turns", type=int, default=5)
    parser.add_argument("--summary-chars", type=int, default=500)
    parser.add_argument("--description", "-d", default="unnamed",
                        help="Experiment description for logging")
    parser.add_argument("--seed-file", default=None,
                        help="Custom seed prompt file (default: built-in benchmark)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (default: ./rfl_autoresearch/run_<timestamp>)")
    parser.add_argument("--log", default=None,
                        help="Results TSV file (default: ./rfl_autoresearch/results.tsv)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use mock output (no Hermes calls) for testing the scoring pipeline")
    args = parser.parse_args()

    # Load seed prompt
    if args.seed_file:
        seed = Path(args.seed_file).read_text().strip()
    else:
        seed = DEFAULT_SEED

    # Output dir
    if args.output:
        output_dir = args.output
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = str(Path(__file__).parent / "run" / timestamp)

    # Results log
    results_file = Path(args.log) if args.log else Path(__file__).parent / "results.tsv"

    config = LoopConfig(
        seed_prompt=seed,
        max_iterations=args.iterations,
        max_context_tokens=args.budget,
        compaction_strategy=args.strategy,
        hermes_model=args.model,
        hermes_provider=args.provider,
        hermes_no_tools=args.no_tools,
        iteration_timeout=args.timeout,
        recent_turns_verbatim=args.recent_turns,
        medium_turns_bullets=args.medium_turns,
        summary_max_chars=args.summary_chars,
        output_dir=output_dir,
        export_format="both",
    )

    print(f"Running experiment: {args.description}")
    print(f"  Strategy: {config.compaction_strategy}")
    print(f"  Iterations: {config.max_iterations}")
    print(f"  Budget: {config.max_context_tokens} tokens")
    print(f"  Output: {output_dir}")
    if args.dry_run:
        print(f"  Mode: DRY RUN (mock output)")
    print()

    if args.dry_run:
        result = _dry_run(config, args.description)
    else:
        result = run_experiment(config, description=args.description)

    print(format_result(result))

    # Log to TSV
    log_result(result, results_file)
    print(f"\nLogged to {results_file}")


if __name__ == "__main__":
    main()
