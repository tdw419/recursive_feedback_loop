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
from .seed_builder import build_audit_seed, detect_mode
from .issue_parser import parse_issues, check_convergence, deduplicate_issues, filter_by_severity
from .report import generate_report, write_report
from .agents import AgentConfig, parse_agent_string, load_agents_file, validate_agents
from .roundtable import RoundTableRunner, RoundTableConfig
from .evolve import EvolveRunner, EvolveConfig
from .build import BuildRunner, BuildConfig


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

    # --- self-audit ---
    sa = sub.add_parser("self-audit", help="Run a self-audit on a codebase using the RFL loop")
    sa.add_argument("path", help="Path to the project to audit")
    sa.add_argument("--diff", action="store_true", default=False,
                     help="Only audit uncommitted changes (default if git detected)")
    sa.add_argument("--full", action="store_true", default=False,
                     help="Audit the entire codebase")
    sa.add_argument("--iterations", "-n", type=int, default=4,
                     help="Max iterations (default: 4)")
    sa.add_argument("--budget", "-b", type=int, default=12000,
                     help="Max context tokens per iteration (default: 12000)")
    sa.add_argument("--severity-threshold", "-S",
                     choices=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                     default="MEDIUM",
                     help="Only report issues at or above this severity (default: MEDIUM)")
    sa.add_argument("--output-report", "-o", default="",
                     help="Output path for the report file (default: ./self_audit_report.md)")
    sa.add_argument("--json", action="store_true", default=False,
                     help="Output JSON instead of markdown")
    sa.add_argument("--model", "-m", help="Hermes model to use")
    sa.add_argument("--provider", help="Hermes provider")
    sa.add_argument("--strategy", "-s",
                     choices=["sliding_window", "rolling_summary", "hierarchical"],
                     default="hierarchical", help="Compaction strategy")
    sa.add_argument("--timeout", "-t", type=int, default=900,
                     help="Per-iteration timeout in seconds (default: 900)")
    sa.add_argument("--workdir", "-w", default="",
                     help="Working directory for Hermes (default: same as PATH)")

    # --- evolve ---
    ev = sub.add_parser("evolve", help="Evolve a prompt by feeding it back into itself")
    ev.add_argument("prompt", nargs="?", help="Seed prompt to evolve")
    ev.add_argument("--from-file", "-f", help="Read seed prompt from file")
    ev.add_argument("--iterations", "-n", type=int, default=5,
                     help="Max iterations (default: 5)")
    ev.add_argument("--branch-every", type=int, default=0,
                     help="Branch into alternatives every N iterations (0=never)")
    ev.add_argument("--branch-on-stagnation", action="store_true", default=True,
                     help="Auto-branch when output stops changing (default: True)")
    ev.add_argument("--no-auto-branch", action="store_false", dest="branch_on_stagnation",
                     help="Disable auto-branching on stagnation")
    ev.add_argument("--temperature", type=float, default=0.7,
                     help="LLM temperature (default: 0.7)")
    ev.add_argument("--max-tokens", type=int, default=4000,
                     help="Max response tokens per iteration (default: 4000)")
    ev.add_argument("--complexity", "-c", default="balanced",
                     choices=["fast", "balanced", "thorough", "auto"],
                     help="Model complexity tier (default: balanced)")
    ev.add_argument("--model", "-m", default=None,
                     help="Override model (e.g. 'openai/glm-5.1')")
    ev.add_argument("--output", "-o", default="",
                     help="Output directory (default: auto-timestamp)")
    ev.add_argument("--no-snapshots", action="store_false", dest="save_snapshots",
                     help="Don't save per-iteration snapshots")

    # --- build ---
    bd = sub.add_parser("build", help="AI autonomously builds a project from an idea")
    bd.add_argument("prompt", nargs="?", help="Project idea / seed prompt")
    bd.add_argument("--from-file", "-f", help="Read seed prompt from file")
    bd.add_argument("--workdir", "-w", default=".",
                     help="Project directory to build in (default: current dir)")
    bd.add_argument("--iterations", "-n", type=int, default=8,
                     help="Max iterations (default: 8)")
    bd.add_argument("--timeout", "-t", type=int, default=900,
                     help="Per-iteration timeout in seconds (default: 900)")
    bd.add_argument("--model", "-m", default=None,
                     help="Hermes model override")
    bd.add_argument("--provider", default=None,
                     help="Hermes provider override")
    bd.add_argument("--retry-provider", default=None,
                     help="Fallback provider when primary times out (e.g. google)")
    bd.add_argument("--retry-model", default=None,
                     help="Fallback model when primary times out (e.g. gemini-2.5-flash)")
    bd.add_argument("--no-explore", action="store_false", dest="explore_enabled",
                     help="Disable [EXPLORE] possibility branching")
    bd.add_argument("--explore-depth", type=int, default=1,
                     help="Possibilities exploration depth (default: 1)")
    bd.add_argument("--output", "-o", default="",
                     help="Output directory for logs (default: auto-timestamp)")

    # --- roundtable ---
    rt = sub.add_parser("roundtable", help="Multi-agent roundtable -- multiple AIs collaborate on a shared conversation")
    rt.add_argument("prompt", nargs="?", help="Seed prompt (or use --from-file)")
    rt.add_argument("--from-file", "-f", help="Read seed prompt from file")
    rt.add_argument("--agent", "-a", action="append", default=[], metavar="SPEC",
                     help="Agent definition: 'name=X,model=Y,role=Z' (repeat for multiple agents)")
    rt.add_argument("--agents-file", "-A", help="YAML file with agent definitions")
    rt.add_argument("--rounds", "-n", type=int, default=5,
                     help="Max rounds (each round = all agents take a turn) (default: 5)")
    rt.add_argument("--budget", "-b", type=int, default=10000,
                     help="Max context tokens per round (default: 10000)")
    rt.add_argument("--strategy", "-s", choices=["sliding_window", "rolling_summary", "hierarchical"],
                     default="hierarchical", help="Compaction strategy (default: hierarchical)")
    rt.add_argument("--timeout", "-t", type=int, default=600,
                     help="Per-agent timeout in seconds (default: 600)")
    rt.add_argument("--max-runtime", type=int, default=7200,
                     help="Max total runtime in seconds (default: 7200)")
    rt.add_argument("--workdir", "-w", help="Working directory for Hermes agents")
    rt.add_argument("--no-tools", action="store_true", default=False,
                     help="Disable Hermes tools for faster text-only iterations")
    rt.add_argument("--output", "-o", default="",
                     help="Output directory (default: ./rfl_roundtable_<timestamp>)")
    rt.add_argument("--synthesis", help="Custom synthesis instruction for agents")
    rt.add_argument("--synthesis-file", help="Read synthesis instruction from file")
    rt.add_argument("--recent-turns", type=int, default=4,
                     help="Recent turns to keep verbatim (default: 4)")
    rt.add_argument("--medium-turns", type=int, default=6,
                     help="Medium turns to keep as bullets (default: 6)")
    rt.add_argument("--summary-chars", type=int, default=600,
                     help="Max chars for summary block (default: 600)")
    rt.add_argument("--compaction-model", help="Separate model for compaction summarization")
    rt.add_argument("--compaction-provider", help="Separate provider for compaction summarization")
    rt.add_argument("--export", choices=["jsonl", "markdown", "both"], default="both",
                     help="Export format (default: both)")

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


def _normalize_issue_path(file_path: str, project_path: Path) -> str:
    """Normalize issue file paths relative to project root.
    
    The LLM may output paths like 'recursive_feedback_loop/config.py' or
    './config.py' or 'config.py' for the same file. This normalizes them.
    """
    if not file_path:
        return file_path
    p = file_path
    # Strip ./ prefix
    if p.startswith("./"):
        p = p[2:]
    # Try to make relative to project name
    proj_name = project_path.name
    # If path starts with project_name/, strip it
    if p.startswith(proj_name + "/"):
        p = p[len(proj_name) + 1:]
    return p


def cmd_self_audit(args) -> int:
    """Run self-audit on a codebase using the RFL loop."""
    project_path = Path(args.path).resolve()
    if not project_path.is_dir():
        print(f"Error: {project_path} is not a directory", file=sys.stderr)
        return 1

    # Determine audit mode
    if args.full:
        mode = "full"
    elif args.diff:
        mode = "diff"
    else:
        mode = detect_mode(project_path)
        print(f"  Auto-detected mode: {mode}")

    # Build the seed prompt
    print(f"  Building {mode} seed for {project_path.name}...")
    try:
        seed_prompt = build_audit_seed(project_path, mode=mode)
    except Exception as e:
        print(f"Error building seed: {e}", file=sys.stderr)
        return 1

    print(f"  Seed prompt: {len(seed_prompt)} chars")

    # Set up workdir
    workdir = args.workdir or str(project_path)

    # Build config for the RFL loop
    config = LoopConfig(
        seed_prompt=seed_prompt,
        mode="oneshot",
        max_iterations=args.iterations,
        max_context_tokens=args.budget,
        compaction_strategy=args.strategy,
        hermes_model=args.model,
        hermes_provider=args.provider,
        hermes_workdir=workdir,
        hermes_no_tools=False,  # self-audit needs tools to read files
        iteration_timeout=args.timeout,
        output_dir="",
        save_snapshots=True,
        export_format="both",
    )

    # Custom synthesis instruction for audit mode
    config.synthesis_instruction = (
        "You are continuing a code audit. You already analyzed the code in the previous iteration.\n\n"
        "YOUR RULES:\n"
        "1. DO NOT repeat or restate anything from the previous iteration. Zero repetition.\n"
        "2. You MUST go deeper. For each issue already identified, either:\n"
        "   - Provide the EXACT fix (complete code, line numbers, imports)\n"
        "   - Identify the ROOT CAUSE (why does this bug exist?)\n"
        "   - Find NEW issues the previous iteration missed\n"
        "3. Verify previous claims against the ACTUAL source code. Correct wrong line numbers.\n"
        "4. Each issue: [SEVERITY] file:line -- description\n"
        "5. Aim for at least 3 completely NEW observations per iteration."
    )

    runner = LoopRunner(config)

    # Handle Ctrl+C
    def handle_signal(sig, frame):
        print("\nStopping self-audit...", file=sys.stderr)
        runner.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"\nSelf-Audit Starting")
    print(f"  Project:  {project_path.name}")
    print(f"  Mode:     {mode}")
    print(f"  Max iterations: {args.iterations}")
    print(f"  Budget:   {args.budget} tokens/iter")
    print(f"  Strategy: {args.strategy}")
    print()

    # Run the loop
    state = runner.run()

    print(f"\nLoop complete: {state.iteration + 1} iterations, {state.elapsed():.1f}s")
    print(f"  Output dir: {state.output_dir}")

    # Parse issues from each iteration's output
    issue_history = []
    all_issues = []
    for turn in state.conversation.turns:
        if turn.role == "assistant":
            iter_issues = parse_issues(turn.content, iteration=turn.iteration)
            # Normalize file paths relative to project root
            for issue in iter_issues:
                issue.file = _normalize_issue_path(issue.file, project_path)
            issue_history.append(iter_issues)
            all_issues.extend(iter_issues)

    if not all_issues:
        print("\nNo issues parsed from LLM output.")
        print("The loop ran but no structured issues were found.")
        print(f"Check the full output at: {state.output_dir}/conversation.md")
        return 0

    # Check convergence
    convergence = check_convergence(issue_history, args.severity_threshold)

    # Deduplicate all issues
    deduped = deduplicate_issues(all_issues)

    # Filter by severity threshold
    filtered = filter_by_severity(deduped, args.severity_threshold)

    # Generate report
    output_format = "json" if args.json else "markdown"
    report_path_str = args.output_report
    if not report_path_str:
        ext = "json" if args.json else "md"
        report_path_str = str(Path(project_path) / f"self_audit_report.{ext}")

    report_path = write_report(
        issues=filtered,
        issue_history=issue_history,
        convergence=convergence,
        project_path=str(project_path),
        output_path=report_path_str,
        output_format=output_format,
        severity_threshold=args.severity_threshold,
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"SELF-AUDIT RESULTS")
    print(f"{'=' * 60}")
    by_sev = {}
    for i in filtered:
        by_sev[i.severity] = by_sev.get(i.severity, 0) + 1
    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        if sev in by_sev:
            print(f"  {sev:10s} {by_sev[sev]}")
    print(f"  {'TOTAL':10s} {len(filtered)}")
    print(f"  Converged: {'Yes' if convergence.converged else 'No'}")
    print(f"\n  Report saved to: {report_path}")
    print(f"  Full output: {state.output_dir}/conversation.md")
    print()

    return 0


def cmd_roundtable(args) -> int:
    """Run a multi-agent roundtable."""
    # --- Resolve agents ---
    agents = []
    if args.agents_file:
        if args.agent:
            print("WARNING: Both --agents-file and --agent specified. Using --agents-file, ignoring --agent.", file=sys.stderr)
        try:
            agents = load_agents_file(args.agents_file)
        except Exception as e:
            print(f"Error loading agents file: {e}", file=sys.stderr)
            return 1
    elif args.agent:
        for spec in args.agent:
            try:
                agents.append(parse_agent_string(spec))
            except ValueError as e:
                print(f"Error parsing agent spec: {e}", file=sys.stderr)
                return 1
    else:
        print("Error: Define agents with --agent or --agents-file", file=sys.stderr)
        print("Example:", file=sys.stderr)
        print('  rfl roundtable "Analyze this code" \\', file=sys.stderr)
        print('    -a "name=auditor,model=claude-sonnet-4,role=Find bugs" \\', file=sys.stderr)
        print('    -a "name=optimizer,model=gemini-2.5-pro,role=Find perf issues"', file=sys.stderr)
        return 1

    if len(agents) < 2:
        print("Warning: roundtable works best with 2+ agents. Only 1 defined.", file=sys.stderr)

    # --- Resolve seed prompt ---
    seed_prompt = ""
    if args.from_file:
        seed_prompt = Path(args.from_file).read_text().strip()
    elif args.prompt:
        seed_prompt = args.prompt
    else:
        print("Error: provide a seed prompt or --from-file", file=sys.stderr)
        return 1

    # --- Validate ---
    warnings = validate_agents(agents)
    for w in warnings:
        print(f"  WARNING: {w}", file=sys.stderr)

    # --- Build config ---
    config = RoundTableConfig(
        agents=agents,
        seed_prompt=seed_prompt,
        max_rounds=args.rounds,
        max_context_tokens=args.budget,
        compaction_strategy=args.strategy,
        iteration_timeout=args.timeout,
        max_runtime_seconds=args.max_runtime,
        hermes_workdir=args.workdir,
        hermes_no_tools=args.no_tools,
        output_dir=args.output,
        synthesis_instruction=args.synthesis if args.synthesis else None,
        synthesis_instruction_file=args.synthesis_file,
        recent_turns_verbatim=args.recent_turns,
        medium_turns_bullets=args.medium_turns,
        summary_max_chars=args.summary_chars,
        compaction_model=args.compaction_model,
        compaction_provider=args.compaction_provider,
        export_format=args.export,
    )

    runner = RoundTableRunner(config)

    def handle_signal(sig, frame):
        print("\nStopping roundtable...", file=sys.stderr)
        runner.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"Roundtable RFL starting")
    print(f"  Agents: {len(agents)}")
    for a in agents:
        print(f"    {a.display()}")
    print(f"  Rounds: {config.max_rounds}")
    print(f"  Strategy: {config.compaction_strategy}")
    print(f"  Token budget: {config.max_context_tokens}")
    print(f"  Timeout: {config.iteration_timeout}s/agent")
    output_dir = config.get_output_dir()
    print(f"  Output: {output_dir}")
    print()

    state = runner.run()

    print()
    print(f"{'=' * 60}")
    print(f"ROUNDTABLE COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Rounds: {state.iteration + 1}")
    print(f"  Total turns: {len(state.conversation)}")
    print(f"  Total tokens: ~{state.conversation.total_tokens_estimate()}")
    print(f"  Time: {state.elapsed():.1f}s")
    print(f"  Agents: {', '.join(state.conversation.agent_names())}")
    print(f"  Output: {state.output_dir}")
    print()

    return 0


def cmd_evolve(args) -> int:
    """Run an evolve loop -- one prompt that keeps feeding itself back."""
    # Resolve seed prompt
    if args.from_file:
        seed = Path(args.from_file).read_text()
    elif args.prompt:
        seed = args.prompt
    else:
        print("Error: provide a prompt or --from-file", file=sys.stderr)
        return 1

    config = EvolveConfig(
        seed_prompt=seed,
        iterations=args.iterations,
        branch_every=args.branch_every,
        branch_on_stagnation=args.branch_on_stagnation,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        complexity=args.complexity,
        model=args.model,
        output_dir=args.output,
        save_snapshots=args.save_snapshots,
    )

    print(f"Evolve starting")
    print(f"  Seed: {len(seed)} chars")
    print(f"  Iterations: {config.iterations}")
    print(f"  Branch every: {config.branch_every or 'off'}")
    print(f"  Auto-branch: {config.branch_on_stagnation}")
    print(f"  Complexity: {config.complexity}")
    print()

    runner = EvolveRunner(config)

    # Handle Ctrl+C
    def handle_signal(sig, frame):
        print("\nStopping evolve...", file=sys.stderr)
        runner.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    state = runner.run()

    print(f"\n{'=' * 60}")
    print(f"EVOLVE COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Iterations: {len(state.turns)}")
    print(f"  Time: {state.elapsed():.1f}s")
    print(f"  Converged at: {state.converged_at or 'did not converge'}")
    print(f"  Final output: {len(state.final_output)} chars")
    branches = sum(1 for t in state.turns if t.branched)
    if branches:
        print(f"  Branched: {branches} iterations")
    print(f"  Output: {state.output_dir}/")
    print()

    # Print the final output
    print("--- FINAL OUTPUT ---")
    print(state.final_output[:2000])
    if len(state.final_output) > 2000:
        print(f"\n... ({len(state.final_output) - 2000} more chars, see {state.output_dir}/final_output.txt)")

    return 0


def cmd_build(args) -> int:
    """Run an autonomous build loop -- AI builds a project from an idea."""
    # Resolve seed prompt
    if args.from_file:
        seed = Path(args.from_file).read_text()
    elif args.prompt:
        seed = args.prompt
    else:
        print("Error: provide a project idea or --from-file", file=sys.stderr)
        return 1

    workdir = str(Path(args.workdir).resolve())
    if not Path(workdir).is_dir():
        print(f"Error: {workdir} is not a directory", file=sys.stderr)
        return 1

    config = BuildConfig(
        seed_prompt=seed,
        workdir=workdir,
        iterations=args.iterations,
        iteration_timeout=args.timeout,
        hermes_model=args.model,
        hermes_provider=args.provider,
        retry_provider=args.retry_provider,
        retry_model=args.retry_model,
        explore_enabled=args.explore_enabled,
        explore_depth=args.explore_depth,
        output_dir=args.output,
    )

    print(f"Build starting")
    print(f"  Seed: {seed[:100]}{'...' if len(seed) > 100 else ''}")
    print(f"  Workdir: {workdir}")
    print(f"  Iterations: {config.iterations}")
    print(f"  Timeout: {config.iteration_timeout}s/iter")
    print(f"  Model: {config.hermes_model or 'default'}")
    print(f"  Provider: {config.hermes_provider or 'default'}")
    if config.retry_provider:
        print(f"  Retry: {config.retry_model or 'default'} via {config.retry_provider}")
    print(f"  Explore: {'enabled' if config.explore_enabled else 'disabled'}")
    print()

    runner = BuildRunner(config)

    def handle_signal(sig, frame):
        print("\nStopping build...", file=sys.stderr)
        runner.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    state = runner.run()

    explored = sum(1 for t in state.turns if t.explored)
    print(f"\n{'=' * 60}")
    print(f"BUILD {'COMPLETE' if state.turns else 'ABORTED'}")
    print(f"{'=' * 60}")
    print(f"  Iterations: {len(state.turns)}")
    print(f"  Time: {state.elapsed():.1f}s")
    if explored:
        print(f"  Explored: {explored} iterations")
    print(f"  Output: {state.output_dir}/")
    print()

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
    elif args.command == "self-audit":
        return cmd_self_audit(args)
    elif args.command == "evolve":
        return cmd_evolve(args)
    elif args.command == "build":
        return cmd_build(args)
    elif args.command == "roundtable":
        return cmd_roundtable(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
