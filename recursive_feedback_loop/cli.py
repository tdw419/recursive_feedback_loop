#!/usr/bin/env python3
"""Recursive Feedback Loop CLI.

Usage:
  rfl run "Initial prompt to start the loop"
  rfl run --from-file seed_prompt.txt
  rfl run "Start thinking about consciousness" --iterations 20 --strategy hierarchical
  rfl run "Explore emergent behavior" --model anthropic/claude-sonnet-4 --budget 12000
"""

import argparse
import sys
import signal
from pathlib import Path

from .config import LoopConfig
from .loop_runner import LoopRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rfl",
        description="Recursive Feedback Loop — AI self-feeding conversation engine",
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- run ---
    run = sub.add_parser("run", help="Run a recursive feedback loop")
    run.add_argument("prompt", nargs="?", help="Seed prompt (or use --from-file)")
    run.add_argument("--from-file", "-f", help="Read seed prompt from file")
    run.add_argument("--mode", "-M", choices=["oneshot", "session"], default="oneshot",
                     help="Execution mode: oneshot (fresh Hermes per iter) or session (persistent tmux)")
    run.add_argument("--iterations", "-n", type=int, default=10, help="Max iterations (default: 10)")
    run.add_argument("--budget", "-b", type=int, default=8000, help="Max context tokens per iteration (default: 8000)")
    run.add_argument("--strategy", "-s", choices=["sliding_window", "rolling_summary", "hierarchical"],
                     default="hierarchical", help="Compaction strategy (default: hierarchical)")
    run.add_argument("--model", "-m", help="Hermes model (e.g. anthropic/claude-sonnet-4)")
    run.add_argument("--provider", help="Hermes provider (e.g. openrouter, anthropic)")
    run.add_argument("--profile", "-p", help="Hermes profile to use")
    run.add_argument("--workdir", "-w", help="Working directory for Hermes")
    run.add_argument("--no-tools", action="store_true", help="Disable Hermes tools for text-only iterations (faster)")
    run.add_argument("--timeout", "-t", type=int, default=600, help="Per-iteration timeout in seconds (default: 600)")
    run.add_argument("--max-runtime", type=int, default=3600, help="Max total runtime in seconds (default: 3600)")
    run.add_argument("--output", "-o", help="Output directory (default: ./rfl_output)")
    run.add_argument("--synthesis", help="Custom synthesis instruction")
    run.add_argument("--synthesis-file", help="Read synthesis instruction from file")
    run.add_argument("--recent-turns", type=int, default=3, help="Recent turns to keep verbatim (default: 3)")
    run.add_argument("--medium-turns", type=int, default=5, help="Medium turns to keep as bullets (default: 5)")
    run.add_argument("--summary-chars", type=int, default=500, help="Max chars for summary block (default: 500)")
    run.add_argument("--export", choices=["jsonl", "markdown", "both"], default="both", help="Export format")
    run.add_argument("--no-snapshots", action="store_true", help="Don't save snapshots")
    run.add_argument("--compaction-model", help="Separate model for compaction summarization")
    run.add_argument("--compaction-provider", help="Separate provider for compaction summarization")

    # --- replay ---
    replay = sub.add_parser("replay", help="Replay a previous loop run from snapshots")
    replay.add_argument("output_dir", help="Output directory from previous run")
    replay.add_argument("--format", choices=["jsonl", "markdown", "both"], default="markdown", help="Export format")

    # --- info ---
    info = sub.add_parser("info", help="Show info about a previous run")
    info.add_argument("output_dir", help="Output directory to inspect")

    return parser


def cmd_run(args) -> int:
    """Execute the run command."""
    if not args.prompt and not args.from_file:
        print("Error: Provide a seed prompt as argument or use --from-file", file=sys.stderr)
        return 1

    config = LoopConfig(
        seed_prompt=args.prompt or "",
        seed_prompt_file=args.from_file,
        mode=args.mode,
        max_iterations=args.iterations,
        max_context_tokens=args.budget,
        compaction_strategy=args.strategy,
        hermes_model=args.model,
        hermes_provider=args.provider,
        hermes_profile=args.profile,
        hermes_workdir=args.workdir,
        hermes_no_tools=args.no_tools,
        iteration_timeout=args.timeout,
        max_runtime_seconds=args.max_runtime,
        output_dir=args.output or "",
        synthesis_instruction=args.synthesis or "",
        synthesis_instruction_file=args.synthesis_file,
        recent_turns_verbatim=args.recent_turns,
        medium_turns_bullets=args.medium_turns,
        summary_max_chars=args.summary_chars,
        export_format=args.export,
        save_snapshots=not args.no_snapshots,
        compaction_model=args.compaction_model,
        compaction_provider=args.compaction_provider,
    )

    runner = LoopRunner(config)

    # Handle Ctrl+C gracefully
    def handle_signal(sig, frame):
        print("\nStopping loop gracefully...", file=sys.stderr)
        runner.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"Recursive Feedback Loop starting")
    print(f"  Mode: {config.mode}")
    print(f"  Strategy: {config.compaction_strategy}")
    print(f"  Token budget: {config.max_context_tokens}")
    print(f"  Max iterations: {config.max_iterations}")
    print(f"  Output: {config.get_output_dir()}")
    print()

    state = runner.run()

    print()
    print(f"Loop complete: {state.iteration + 1} iterations, {state.elapsed():.1f}s")
    print(f"  Total turns: {len(state.conversation)}")
    print(f"  Total tokens: ~{state.conversation.total_tokens_estimate()}")
    print(f"  Output dir: {state.output_dir}")
    return 0


def cmd_replay(args) -> int:
    """Replay a previous run."""
    import json

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: {output_dir} does not exist", file=sys.stderr)
        return 1

    snap_dir = output_dir / "snapshots"
    if not snap_dir.exists():
        print(f"Error: No snapshots in {output_dir}", file=sys.stderr)
        return 1

    # Load latest snapshot
    snaps = sorted(snap_dir.glob("iter_*.json"))
    if not snaps:
        print("No snapshots found", file=sys.stderr)
        return 1

    latest = snaps[-1]
    data = json.loads(latest.read_text())
    print(f"Latest snapshot: {latest.name}")
    print(f"  Iteration: {data['iteration']}")
    print(f"  Total tokens: ~{data['total_tokens_estimate']}")
    print(f"  Turns: {len(data['turns'])}")
    return 0


def cmd_info(args) -> int:
    """Show info about a previous run."""
    import json

    output_dir = Path(args.output_dir)
    log_file = output_dir / "loop_log.jsonl"

    if not log_file.exists():
        print(f"No log found at {log_file}", file=sys.stderr)
        return 1

    events = []
    with open(log_file) as f:
        for line in f:
            events.append(json.loads(line.strip()))

    if not events:
        print("Empty log", file=sys.stderr)
        return 1

    start = events[0]
    end = [e for e in events if e.get("event") == "loop_end"]
    end = end[-1] if end else events[-1]

    print(f"Loop Run Info: {output_dir}")
    print(f"  Started: {start['timestamp']}")
    if "loop_end" in [e["event"] for e in events]:
        print(f"  Ended: {end['timestamp']}")
        if "data" in end:
            d = end["data"]
            print(f"  Iterations: {d.get('total_iterations', '?')}")
            print(f"  Total tokens: ~{d.get('total_tokens', '?')}")
            print(f"  Elapsed: {d.get('elapsed_seconds', '?')}s")
    print(f"  Log entries: {len(events)}")

    # Show compaction efficiency
    compactions = [e for e in events if e.get("event") == "compacted"]
    if compactions:
        print(f"\n  Compaction stats:")
        for c in compactions:
            d = c.get("data", {})
            orig = d.get("original_tokens", "?")
            comp = d.get("compacted_tokens", "?")
            ratio = f"{comp/orig:.1%}" if isinstance(orig, int) and isinstance(comp, int) and orig > 0 else "?"
            print(f"    iter {c['iteration']}: {orig} -> {comp} tokens ({ratio})")

    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        return cmd_run(args)
    elif args.command == "replay":
        return cmd_replay(args)
    elif args.command == "info":
        return cmd_info(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
