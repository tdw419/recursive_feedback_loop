#!/usr/bin/env python3
"""Recursive Feedback Loop CLI.

Usage:
  rfl run "Initial prompt to start the loop"
  rfl run --from-file seed_prompt.txt
  rfl run --template audit -w /path/to/project -p description="My project"
  rfl run --template feature -w /path/to/project -p feature_name="infinite map"
  rfl templates                    # list available templates
  rfl templates audit              # show template details
  rfl list                         # show running instances
"""

import argparse
import sys
import signal
from pathlib import Path

from .config import LoopConfig, find_running_rfl_instances
from .loop_runner import LoopRunner
from .templates import load_template, list_templates, apply_template


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rfl",
        description="Recursive Feedback Loop — AI self-feeding conversation engine",
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- run ---
    run = sub.add_parser("run", help="Run a recursive feedback loop")
    run.add_argument("prompt", nargs="?", help="Seed prompt (or use --from-file or --template)")
    run.add_argument("--from-file", "-f", help="Read seed prompt from file")
    run.add_argument("--template", "-T", help="Use a built-in template (audit, feature, bug-hunt, spec-review, security, creative)")
    run.add_argument("--param", "-p", action="append", default=[], metavar="KEY=VALUE",
                     help="Fill a template placeholder: -p name=value (can repeat)")
    run.add_argument("--mode", "-M", choices=["oneshot", "session"], default=None,
                     help="Execution mode: oneshot (fresh Hermes per iter) or session (persistent tmux)")
    run.add_argument("--iterations", "-n", type=int, default=None, help="Max iterations (default: 10)")
    run.add_argument("--budget", "-b", type=int, default=None, help="Max context tokens per iteration (default: 8000)")
    run.add_argument("--strategy", "-s", choices=["sliding_window", "rolling_summary", "hierarchical"],
                     default=None, help="Compaction strategy (default: hierarchical)")
    run.add_argument("--model", "-m", help="Hermes model (e.g. anthropic/claude-sonnet-4)")
    run.add_argument("--provider", help="Hermes provider (e.g. openrouter, anthropic)")
    run.add_argument("--profile", "-P", help="Hermes profile to use")
    run.add_argument("--workdir", "-w", help="Working directory for Hermes")
    run.add_argument("--no-tools", action="store_true", default=False,
                     help="Disable Hermes tools for text-only iterations (faster)")
    run.add_argument("--timeout", "-t", type=int, default=None, help="Per-iteration timeout in seconds (default: 600)")
    run.add_argument("--max-runtime", type=int, default=None, help="Max total runtime in seconds (default: 3600)")
    run.add_argument("--output", "-o", help="Output directory (default: ./rfl_output_<timestamp>)")
    run.add_argument("--synthesis", help="Custom synthesis instruction")
    run.add_argument("--synthesis-file", help="Read synthesis instruction from file")
    run.add_argument("--recent-turns", type=int, default=None, help="Recent turns to keep verbatim (default: 3)")
    run.add_argument("--medium-turns", type=int, default=None, help="Medium turns to keep as bullets (default: 5)")
    run.add_argument("--summary-chars", type=int, default=None, help="Max chars for summary block (default: 500)")
    run.add_argument("--export", choices=["jsonl", "markdown", "both"], default=None, help="Export format")
    run.add_argument("--no-snapshots", action="store_true", default=False, help="Don't save snapshots")
    run.add_argument("--compaction-model", help="Separate model for compaction summarization")
    run.add_argument("--compaction-provider", help="Separate provider for compaction summarization")
    run.add_argument("--tmux-session", help="Custom tmux session name (session mode, auto-derived if not set)")

    # --- replay ---
    replay = sub.add_parser("replay", help="Replay a previous loop run from snapshots")
    replay.add_argument("output_dir", help="Output directory from previous run")
    replay.add_argument("--format", choices=["jsonl", "markdown", "both"], default="markdown", help="Export format")

    # --- info ---
    info = sub.add_parser("info", help="Show info about a previous run")
    info.add_argument("output_dir", help="Output directory to inspect")

    # --- list ---
    lst = sub.add_parser("list", help="Show running RFL instances")

    # --- templates ---
    tmpl = sub.add_parser("templates", help="List or show RFL templates")
    tmpl.add_argument("name", nargs="?", help="Template name to show details for")

    return parser


def _parse_params(param_list: list) -> dict:
    """Parse -p key=value pairs into a dict."""
    params = {}
    for item in param_list:
        if "=" in item:
            key, value = item.split("=", 1)
            params[key.strip()] = value.strip()
        else:
            print(f"Warning: ignoring malformed param '{item}' (expected key=value)", file=sys.stderr)
    return params


def cmd_run(args) -> int:
    """Execute the run command."""
    seed_prompt = ""
    template_overrides = {}

    # --- Load template if specified ---
    if args.template:
        tmpl = load_template(args.template)
        if not tmpl:
            print(f"Error: Unknown template '{args.template}'", file=sys.stderr)
            print("Run 'rfl templates' to see available templates.", file=sys.stderr)
            return 1

        params = _parse_params(args.param)
        result = apply_template(tmpl, params)
        seed_prompt = result["seed_prompt"]
        template_overrides = result["config_overrides"]

        # Check for unfilled placeholders
        import re
        unfilled = re.findall(r'\{\{(\w+)\}\}', seed_prompt)
        if unfilled:
            print(f"Warning: Unfilled placeholders in template: {', '.join(unfilled)}", file=sys.stderr)
            print("Use -p key=value to fill them. Example:", file=sys.stderr)
            for p in unfilled[:3]:
                print(f"  -p {p}=\"your value here\"", file=sys.stderr)
    elif args.from_file:
        seed_prompt = Path(args.from_file).read_text().strip()
    elif args.prompt:
        seed_prompt = args.prompt
    else:
        print("Error: Provide a seed prompt, --from-file, or --template", file=sys.stderr)
        return 1

    # --- Build config: CLI args override template defaults ---
    config = LoopConfig(
        seed_prompt=seed_prompt,
        mode=args.mode or template_overrides.get("mode", "oneshot"),
        max_iterations=args.iterations if args.iterations is not None else template_overrides.get("iterations", 10),
        max_context_tokens=args.budget if args.budget is not None else template_overrides.get("budget", 8000),
        compaction_strategy=args.strategy or template_overrides.get("strategy", "hierarchical"),
        hermes_model=args.model,
        hermes_provider=args.provider,
        hermes_profile=args.profile,
        hermes_workdir=args.workdir,
        hermes_no_tools=args.no_tools or template_overrides.get("no_tools", False),
        iteration_timeout=args.timeout if args.timeout is not None else template_overrides.get("timeout", 600),
        max_runtime_seconds=args.max_runtime or 3600,
        output_dir=args.output or "",
        synthesis_instruction=args.synthesis if args.synthesis else None,
        synthesis_instruction_file=args.synthesis_file,
        recent_turns_verbatim=args.recent_turns or 3,
        medium_turns_bullets=args.medium_turns or 5,
        summary_max_chars=args.summary_chars or 500,
        export_format=args.export or "both",
        save_snapshots=not args.no_snapshots,
        compaction_model=args.compaction_model,
        compaction_provider=args.compaction_provider,
        tmux_session_name=args.tmux_session or "",
    )

    runner = LoopRunner(config)

    # Handle Ctrl+C gracefully
    def handle_signal(sig, frame):
        print("\nStopping loop gracefully...", file=sys.stderr)
        runner.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"Recursive Feedback Loop starting")
    if args.template:
        print(f"  Template: {args.template}")
    print(f"  Mode: {config.mode}")
    print(f"  Strategy: {config.compaction_strategy}")
    print(f"  Token budget: {config.max_context_tokens}")
    print(f"  Max iterations: {config.max_iterations}")
    output_dir = config.get_output_dir()
    print(f"  Output: {output_dir}")
    if config.mode == "session":
        print(f"  Tmux session: {config.get_tmux_session_name()}")
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


def cmd_list(args) -> int:
    """Show running RFL instances."""
    instances = find_running_rfl_instances()

    if not instances:
        print("No running RFL instances found.")
        return 0

    print(f"Running RFL instances ({len(instances)}):\n")
    for inst in instances:
        print(f"  PID {inst.get('pid', '?')}  run={inst.get('run_id', '?')}")
        print(f"    Output:    {inst.get('output_dir', '?')}")
        if inst.get("workdir"):
            print(f"    Workdir:   {inst['workdir']}")
        print(f"    Mode:      {inst.get('mode', '?')}")
        if inst.get("tmux_session"):
            print(f"    Tmux:      {inst['tmux_session']}")
        print(f"    Started:   {inst.get('started_at', '?')}")
        print()

    # Check for workdir conflicts
    workdirs = {}
    for inst in instances:
        wd = inst.get("workdir", "")
        if wd:
            workdirs.setdefault(wd, []).append(inst)
    conflicts = {wd: insts for wd, insts in workdirs.items() if len(insts) > 1}
    if conflicts:
        print("CONFLICTS (same workdir, multiple instances):")
        for wd, insts in conflicts.items():
            pids = [str(i["pid"]) for i in insts]
            print(f"  {wd}: PIDs {', '.join(pids)}")
        print()

    return 0


def cmd_templates(args) -> int:
    """List or show RFL templates."""
    if args.name:
        # Show specific template
        tmpl = load_template(args.name)
        if not tmpl:
            print(f"Unknown template: {args.name}", file=sys.stderr)
            print("Run 'rfl templates' to see available templates.", file=sys.stderr)
            return 1

        print(f"Template: {tmpl.name}")
        print(f"Description: {tmpl.description}")
        print(f"Source: {tmpl.source}")
        print()
        if tmpl.mode:
            print(f"  mode:       {tmpl.mode}")
        if tmpl.iterations:
            print(f"  iterations: {tmpl.iterations}")
        if tmpl.budget:
            print(f"  budget:     {tmpl.budget}")
        if tmpl.strategy:
            print(f"  strategy:   {tmpl.strategy}")
        if tmpl.no_tools is not None:
            print(f"  no_tools:   {tmpl.no_tools}")
        if tmpl.timeout:
            print(f"  timeout:    {tmpl.timeout}")
        if tmpl.placeholders:
            print(f"\n  Placeholders:")
            for p in sorted(tmpl.placeholders):
                print(f"    {{{{{p}}}}}")
        print()
        print("Seed prompt:")
        print("-" * 40)
        print(tmpl.seed_prompt)
        print("-" * 40)
        print()
        print(f"Usage:")
        print(f"  rfl run --template {tmpl.name} -w /path/to/project \\")
        if tmpl.placeholders:
            for p in sorted(tmpl.placeholders)[:3]:
                print(f'    -p {p}="value" \\')
        print(f"    -o ./my_{tmpl.name}_run")
        return 0

    # List all templates
    templates = list_templates()
    if not templates:
        print("No templates found.")
        return 0

    print(f"Available templates ({len(templates)}):\n")
    for tmpl in sorted(templates, key=lambda t: t.name):
        placeholders = ""
        if tmpl.placeholders:
            placeholders = f"  params: {', '.join('{' + p + '}' for p in sorted(tmpl.placeholders))}"
        print(f"  {tmpl.name:15s} {tmpl.description}")
        if placeholders:
            print(f"  {'':15s} {placeholders}")
        print(f"  {'':15s} source: {tmpl.source}")
    print()
    print("Usage:")
    print("  rfl run --template <name> -w /path/to/project -p key=value")
    print("  rfl templates <name>  # show details")
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
    elif args.command == "list":
        return cmd_list(args)
    elif args.command == "templates":
        return cmd_templates(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
