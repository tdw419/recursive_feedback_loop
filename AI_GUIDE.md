# RFL Guide for AI Agents

## The Problem RFL Solves

When you ask an AI to analyze a codebase, it does one pass and stops. It finds surface-level issues, misses deeper bugs, and never corrects its own mistakes. The most critical findings often require thinking about what you already found — "that bug I saw in iteration 1 actually causes the crash in iteration 3."

RFL (Recursive Feedback Loop) fixes this by running the AI multiple times, feeding each output back as the next input. Each iteration sees what the previous one found, corrects its mistakes, and goes deeper. The last iteration consistently finds the worst bugs.

## Three Loop Topologies

RFL has three commands, each a different loop shape for a different problem:

| Command | Context flow | Compaction | Tools | Use for |
|---------|-------------|------------|-------|---------|
| `rfl run` | seed → compact → synthesize | Lossy (hierarchical) | Hermes | Analysis, audits, debugging |
| `rfl evolve` | seed → refine → refine | Lossless (full output) | None (model_choice) | Prompt refinement, spec writing |
| `rfl build` | seed → build → build | Lossless + project state | Hermes | Autonomous project creation |

### `rfl run` — The Original Loop

```
1. Seed prompt --> Hermes (AI agent) --> Response #0
2. Compact Response #0 into key points
3. Feed compacted context + "go deeper" instruction --> Hermes --> Response #1
4. Compact Response #0 + #1
5. Feed compacted context --> Hermes --> Response #2
... repeat N times
6. Export all responses to conversation.md
```

**Compaction** is the key. Without it, context doubles each iteration (each output becomes next input). RFL compresses old turns into summaries, medium turns into bullet points, and keeps only recent turns verbatim. This keeps context within a fixed token budget regardless of how many iterations you run.

**Three compaction strategies:**
- `hierarchical` (default, best): distant=summary, medium=bullets, recent=verbatim
- `rolling_summary`: LLM-summarizes old turns, keeps recent verbatim
- `sliding_window`: keeps last N turns, drops everything else

## Installation

```bash
cd ~/zion/projects/recursive_feedback_loop && pip install -e . --break-system-packages
```

Requires `hermes` CLI installed and in PATH. Verify: `which hermes`

## The Fastest Way: Templates

Templates bundle a pre-written seed prompt with tuned config defaults. Instead of assembling flags and writing seed prompts from scratch, pick a template and fill in the blanks.

### Available Templates

| Name | Purpose | Mode | Iters | Budget | Tools |
|------|---------|------|-------|--------|-------|
| `audit` | Deep codebase audit — bugs, architecture, gaps | oneshot | 5 | 12K | yes |
| `feature` | Design + build + test a new feature | oneshot | 5 | 12K | yes |
| `bug-hunt` | Track down a specific bug with hypotheses | oneshot | 4 | 10K | yes |
| `security` | Vulnerability review (injection, auth, crypto) | oneshot | 4 | 12K | yes |
| `spec-review` | Review a spec for gaps and contradictions | oneshot | 3 | 10K | no |
| `creative` | Open-ended philosophical/creative exploration | session | 5 | 8K | no |

### How to Use Templates

```bash
# See what's available
rfl templates

# See a specific template's seed prompt and what placeholders it needs
rfl templates audit

# Run a template — fill placeholders with -p key=value
rfl run --template audit -w /path/to/project \
  -p description="Geometry OS — pixel OS with RISC-V interpreter" \
  -p source_dirs="src/core, src/riscv, src/assembler"
```

Every template has `{{placeholder}}` variables in its seed prompt. You fill them with `-p key=value`. If you forget one, RFL warns you but still runs.

**CLI flags override template defaults** — so `--template audit -n 3` gives you the audit seed prompt but only 3 iterations instead of 5.

### Example: Audit a Project

```bash
rfl run --template audit \
  -w ~/zion/projects/geometry_os/geometry_os \
  -p description="Geometry OS — pixel operating system with 100 assembler mnemonics, RISC-V interpreter, VFS" \
  -p source_dirs="src/core, src/riscv, src/assembler, src/vfs" \
  -o ./geometry_os_audit
```

What happens across iterations:
- Iteration 0: Surface scan, reads source files, flags obvious issues
- Iteration 1: Corrects iteration 0 mistakes, verifies claims against code, finds deeper bugs
- Iteration 2-4: Progressive deepening — finds the bugs that require understanding earlier findings
- Output: `./geometry_os_audit/conversation.md`

### Example: Hunt a Bug

```bash
rfl run --template bug-hunt \
  -w ~/zion/projects/my_project \
  -p bug_description="Segfault when parsing empty config file" \
  -p symptom="Application crashes with SIGSEGV on startup if config.yml is empty" \
  -p reproduction_steps="touch config.yml && ./my_app" \
  -p expected_behavior="Graceful error: 'Config file is empty'" \
  -p actual_behavior="Segfault in config_parser.rs line 47" \
  -p source_dirs="src/config, src/main" \
  -o ./bug_investigation
```

What happens:
- Iteration 0: Reads the relevant source, traces control flow, forms first hypothesis
- Iteration 1: Tests hypothesis against code, corrects if wrong, digs deeper
- Iteration 2-3: Narrows to root cause, writes fix, checks for similar bugs elsewhere

### Example: Build a Feature

```bash
rfl run --template feature \
  -w ~/zion/projects/my_project \
  -p feature_name="infinite procedural map" \
  -p description="Pixel OS with 100 opcodes, needs tile-based terrain generation" \
  -p existing_code="src/asm/opcodes.rs has MATH/COMPARE/JUMP ops. src/canvas.rs handles pixel buffer." \
  -p constraints="Must use existing opcodes only. No new host-language code. Budget: 512 bytes assembled." \
  -p source_dirs="src/asm, src/canvas, src/kernel" \
  -o ./feature_infinite_map
```

### Example: Security Review

```bash
rfl run --template security \
  -w ~/zion/projects/web_app \
  -p description="FastAPI web app with JWT auth, PostgreSQL backend, file upload API" \
  -p source_dirs="src/routes, src/auth, src/models, src/upload" \
  -p threat_model="External attacker with no auth. Focus on injection, auth bypass, file upload abuse." \
  -o ./security_review
```

## The Manual Way (No Template)

If no template fits your task, write your own seed prompt.

```bash
# Inline prompt
rfl run "Analyze the error handling patterns in this codebase" \
  -n 5 -b 12000 -w /path/to/project

# From a file (better for long seeds)
cat > /tmp/my_seed.md << 'EOF'
Your custom seed prompt here.
Be specific about what you want the AI to do.
EOF
rfl run --from-file /tmp/my_seed.md -n 5 -b 12000 -w /path/to/project -o ./my_run
```

## Two Execution Modes

### Oneshot (default, recommended)

Spawns a fresh Hermes subprocess per iteration. The compactor manages context between iterations. Most reliable — if one iteration crashes, the next one starts clean.

Best for: audits, feature builds, bug hunts, anything with tool calls.

### Session (tmux)

Keeps one Hermes process alive in a tmux pane. Hermes retains full conversation history internally — no compaction needed. But tmux is fragile: long tool-call chains time out, UI changes break response extraction.

Best for: creative exploration, philosophical prompts, short prompts without tools.

### `rfl evolve` — Lossless Refinement

```bash
rfl evolve "Write a clear API specification for a pixel encoding system" -n 5 -c balanced
```

One prompt feeds itself back continuously. Each iteration takes the previous OUTPUT as input, adds a refinement instruction, and sends to an LLM via model_choice (no Hermes subprocess). No compaction — the full refined text becomes the next input.

**Key features:**
- Auto-branching on stagnation (when output stops changing, explores alternatives)
- Configurable branch frequency (`--branch-every N`)
- Convergence detection (stops early if output stabilizes)

**Options:**

| Flag | Short | Default | What it does |
|------|-------|---------|-------------|
| `--iterations` | `-n` | 5 | Max iterations |
| `--branch-every` | | 0 | Branch every N iterations (0=never) |
| `--branch-on-stagnation` | | True | Auto-branch when output stops changing |
| `--temperature` | | 0.7 | LLM temperature |
| `--max-tokens` | | 4000 | Max response tokens per iteration |
| `--complexity` | `-c` | balanced | model_choice tier: fast, balanced, thorough, auto |
| `--model` | `-m` | default | Override model (e.g. 'openai/glm-5.1') |
| `--output` | `-o` | auto-timestamp | Output directory |

**Output:** `conversation.md` (all iterations), `final_output.txt` (last iteration), `state.json` (machine-readable).

### `rfl build` — Autonomous Project Builder

```bash
rfl build "Build a pixel-based byte encoder in Rust with 4 strategies" -w ./my_project -n 8
```

The AI designs, builds, tests, and iterates on a real project. Each iteration the AI sees what it produced last time, what files exist in the project, and continues building however it wants. No prescriptive instructions — the AI decides what to do next.

**Flow:**
1. Iteration 0: seed prompt → Hermes (with tools) → design + start building
2. Iteration N: project state + last output → Hermes → continue building
3. Between iterations: detect `[EXPLORE: question]` triggers, run ai_possibilities if found
4. When done: AI writes `[DONE]` and the loop stops

**Different from `run`:** No compaction. No prescriptive "go deeper" synthesis. The AI has full creative control.

**Different from `evolve`:** Uses Hermes subprocess (has tools). Reads actual project state between iterations. Builds a real project, not just refined text.

**Options:**

| Flag | Short | Default | What it does |
|------|-------|---------|-------------|
| `--workdir` | `-w` | current dir | Project directory to build in |
| `--iterations` | `-n` | 8 | Max iterations |
| `--timeout` | `-t` | 900 | Seconds per iteration |
| `--model` | `-m` | default | Hermes model override |
| `--provider` | | default | Hermes provider override |
| `--no-explore` | | off | Disable [EXPLORE] possibility branching |
| `--explore-depth` | | 1 | Possibilities exploration depth |
| `--output` | `-o` | auto-timestamp | Output directory for logs |
| `--from-file` | `-f` | | Read seed prompt from file |

**What the AI sees each iteration:**
- The original seed prompt
- Current project state (file listing with sizes, git log, git diff stats)
- Its own previous output (last iteration's full response)
- Any exploration results from [EXPLORE] triggers

**Escape hatches:**
- `[EXPLORE: question]` — AI is stuck, wants alternatives. The loop runs ai_possibilities and feeds branches back.
- `[DONE]` — AI declares the project complete. Loop stops.

**Output:** `conversation.md` (all iterations), `build_log.jsonl` (per-iteration stats), `snapshots/` (per-iteration state).

**Known issues:**
- Hermes file tools resolve relative to its auto-detected project root, not the subprocess cwd. The prompt instructs Hermes to use absolute paths and terminal commands, but complex projects may still have path issues.
- Complex builds (Rust, C++) may need 600-900s per iteration. The first iteration is often the longest (scaffolding + compilation).
- If an iteration times out, files it created survive. The next iteration picks up where it left off via project state injection.

## Key Flags Reference

| Flag | Short | Default | What it does |
|------|-------|---------|-------------|
| `--iterations` | `-n` | 10 | Max iterations. 3-5 is usually enough. |
| `--budget` | `-b` | 8000 | Max context tokens per iteration. 12K for code-heavy tasks. |
| `--workdir` | `-w` | cwd | Where Hermes runs. The AI can read/write files here. |
| `--output` | `-o` | auto-timestamped | Where RFL writes logs, snapshots, exports. |
| `--no-tools` | | off | Disable Hermes tools for iterations > 0. Faster text-only synthesis. |
| `--timeout` | `-t` | 600 | Seconds per iteration. 900 for tool-heavy seeds. |
| `--strategy` | `-s` | hierarchical | Compaction strategy. |
| `--mode` | `-M` | oneshot | oneshot or session. |
| `--template` | `-T` | | Template name. |
| `--param` | `-p` | | Fill template placeholder: `-p key=value`. Repeatable. |
| `--model` | `-m` | default | Hermes model override. |
| `--synthesis` | | default | Custom "go deeper" instruction. |

## What Gets Output

Every run creates an output directory:

```
rfl_output_20260414_171103/
  .rfl.lock             # PID lockfile (auto-managed, deleted on exit)
  loop_log.jsonl        # Timestamped event log (start, compaction, iteration, end)
  snapshots/
    iter_000.json       # Full conversation state after each iteration
    iter_001.json
    ...
  conversation.jsonl    # Machine-readable: one JSON object per turn
  conversation.md       # Human-readable: the main deliverable
```

**Read `conversation.md`** for the results. Each iteration is a separate section with headers.

```bash
# Check a completed run
rfl info ./rfl_output_20260414_171103

# See compaction stats
rfl info ./rfl_output_20260414_171103
# Shows: iter 0: 3200 -> 1200 tokens (37.5%)
```

## Running Multiple Instances in Parallel

Each RFL run is isolated in its own output directory. You can run multiple at the same time.

**Automatically isolated per-run:**
- Output directory (auto-timestamped, unique per run)
- Tmux session (unique name per run)
- PID lockfile (prevents two runs claiming the same output dir)

**NOT isolated — your responsibility:**
- `--workdir`. Two instances pointing at the same workdir means two AIs modifying the same files simultaneously. RFL warns you, but doesn't block you.

**Safe parallel patterns:**
```bash
# Different projects (naturally isolated)
rfl run --template audit -w /path/to/projectA -o ./audit_a &
rfl run --template audit -w /path/to/projectB -o ./audit_b &

# Same project, different branches (git isolates file changes)
cd /path/to/project
git checkout main && git checkout -b rfl_audit
rfl run --template audit -w . -o ./audit_1 &
git checkout main && git checkout -b rfl_security
rfl run --template security -w . -o ./security_1 &

# Check what's running
rfl list
```

If two instances collide on the same workdir, `rfl list` will flag it:
```
CONFLICTS (same workdir, multiple instances):
  /path/to/project: PIDs 12345, 12346
```

## Creating Custom Templates

Drop a directory in `~/.config/rfl/templates/<name>/` with two files:

**template.yaml** — config defaults:
```yaml
name: my-template
description: What this template does
mode: oneshot
iterations: 3
budget: 10000
strategy: hierarchical
no_tools: false
timeout: 600
```

**seed.md** — seed prompt with `{{placeholders}}`:
```markdown
Your seed prompt here.

Project: {{project_name}}
Focus: {{focus}}
```

Use it: `rfl run --template my-template -w . -p project_name="foo" -p focus="perf"`

User templates override built-in templates with the same name.

## Common Mistakes

1. **Forgetting `--workdir`** — The AI needs `-w` to read project files. Without it, Hermes runs in cwd and can't find source.

2. **Session mode for tool-heavy tasks** — Use oneshot. Session mode times out when Hermes makes long tool-call chains.

3. **Parallel instances, same workdir** — Both AIs modify the same files. Use `rfl list` to check. Use git branches or different directories.

4. **Too many iterations** — 3-5 is the sweet spot. Beyond that, diminishing returns. Each iteration costs an LLM call.

5. **Budget too low** — Default is 8K. For code-heavy audits, use 12K. Below 6K, the compactor loses too much context and the AI forgets what it found.

6. **Not using templates** — Templates encode proven patterns. The audit template's iteration-by-iteration deepening was validated through experiments. Use them instead of writing seed prompts from scratch.

## Self-Audit Pattern: RFL Analyzing Itself

The RFL can be pointed at its own source code to find bugs. This is uniquely effective because the AI already understands the architecture and can reason about failure modes precisely.

**Full case study:** `CASE_STUDY_SELF_AUDIT.md`

```bash
# Point the RFL at its own source
rfl run --template audit \
  -w ~/zion/projects/recursive_feedback_loop \
  -p description="RFL — recursive feedback loop tool, ~2K LOC Python, 6 source files" \
  -p source_dirs="recursive_feedback_loop" \
  -o ./rfl_self_audit
```

**What it found (5 iterations, 20.5 min):**
- 3 CRITICAL bugs (synthesis instruction silently killed, output cleaner returning garbage, crash on empty files)
- 5 HIGH bugs (O(n^2) compaction, wrapper overhead, bullet text mangling, deduper gaps)
- 14 additional MEDIUM/LOW issues

**Key insight:** The worst bug (cli.py silently replacing the synthesis prompt with "") was found in iteration 2, not iteration 0. Iterations 0-1 exhausted the obvious algorithmic targets, forcing iteration 2 to look at argument parsing -- the boring glue code where the bug lived.

**All 8 CRITICAL/HIGH bugs were fixed in commit a0270bf.**

## Programmatic Usage

```python
from recursive_feedback_loop.config import LoopConfig
from recursive_feedback_loop.loop_runner import LoopRunner

config = LoopConfig(
    seed_prompt="Explore the concept of infinity",
    max_iterations=5,
    compaction_strategy="hierarchical",
    max_context_tokens=10000,
    hermes_workdir="/path/to/project",
)

runner = LoopRunner(config)
state = runner.run()

for turn in state.conversation.turns:
    print(f"[{turn.role} iter {turn.iteration}] {turn.content[:100]}...")

print(f"Results: {state.output_dir / 'conversation.md'}")
```

## Project Layout

```
~/zion/projects/recursive_feedback_loop/
  recursive_feedback_loop/
    cli.py              # CLI entry point
    config.py           # LoopConfig, concurrency, lockfile logic
    loop_runner.py      # Core orchestration for `rfl run` (compaction loop)
    compaction.py       # Three compaction strategies
    session_mode.py     # Persistent tmux session (session mode)
    session_reader.py   # Hermes JSONL parser
    templates.py        # Template loader, placeholder filling
    templates/          # Built-in templates (audit, feature, etc.)
    evolve.py           # `rfl evolve` — lossless refinement loop
    build.py            # `rfl build` — autonomous project builder
    seed_builder.py     # Self-audit seed prompt builder
    issue_parser.py     # Parse structured issues from LLM output
    report.py           # Self-audit report generation
  tests/                # 33 tests
  rfl_autoresearch/     # Automated tuning harness
  CASE_STUDY_SELF_AUDIT.md  # How the RFL found 8 bugs in itself
  AI_GUIDE.md           # This file
```
