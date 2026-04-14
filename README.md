# Recursive Feedback Loop

AI self-feeding conversation engine. Give it a seed prompt, and it recursively feeds its own output back as the next input — with intelligent compaction to prevent context explosion.

## How It Works

```
Seed Prompt → Hermes → Response
                              ↓
              Compact History ←┘
                     ↓
              Synthesis Prompt (compacted context + instruction)
                     ↓
              Hermes → Response
                     ↓
              Compact History ←┘
                     ↓
              ... repeat N times
```

Each iteration:
1. Takes the full conversation history so far
2. **Compacts** it to fit within a token budget (the key innovation)
3. Wraps it in a synthesis instruction ("continue developing your thoughts")
4. Feeds it back into Hermes as a new prompt
5. Captures the response and adds it to the history

## The Compaction Problem

Without compaction, context grows exponentially — each iteration's output becomes part of the next iteration's input. After N iterations with average response size R, you get O(R × 2^N) tokens. That blows up fast.

### Three Compaction Strategies

| Strategy | How it works | Trade-off |
|---|---|---|
| **sliding_window** | Keep last N turns verbatim, drop rest | Simple, but loses all history |
| **rolling_summary** | LLM-summarize old turns, keep recent verbatim | Preserves themes, needs extra LLM call |
| **hierarchical** | Old→summary, medium→bullets, recent→verbatim | Best balance (default) |

### Hierarchical Compaction (default)

Three tiers of detail:
- **DISTANT CONTEXT**: A compressed paragraph summarizing early conversation
- **RECENT CONTEXT (condensed)**: Bullet points of medium-age turns
- **LATEST TURNS (verbatim)**: Full text of the most recent turns

This gives the AI enough history to maintain thematic continuity without blowing the token budget.

## Installation

```bash
cd ~/zion/projects/recursive_feedback_loop
pip install -e . --break-system-packages
```

Requires `hermes` CLI installed and configured.

## Usage

### Basic
```bash
rfl run "Start exploring the nature of consciousness"
```

### With options
```bash
rfl run "Explore emergent behavior in complex systems" \
  --iterations 20 \
  --strategy hierarchical \
  --budget 12000 \
  --model anthropic/claude-sonnet-4
```

### From file
```bash
rfl run --from-file seed_prompt.txt --iterations 50 --output ./my_loop_run
```

### Custom synthesis instruction
```bash
rfl run "Investigate the relationship between recursion and self-awareness" \
  --synthesis "You are going deeper. Each iteration, find a new dimension of the topic that you haven't explored yet. Push beyond obvious observations."
```

### Inspect a previous run
```bash
rfl info ./rfl_output
rfl replay ./rfl_output
```

## CLI Reference

```
rfl run [PROMPT] [OPTIONS]

Positional:
  prompt                    Seed prompt text

Seed:
  --from-file, -f FILE      Read seed prompt from file

Loop Control:
  --iterations, -n N        Max iterations (default: 10)
  --timeout, -t SECS        Per-iteration timeout (default: 300)
  --max-runtime SECS        Total runtime limit (default: 3600)

Compaction:
  --strategy, -s STRATEGY   sliding_window | rolling_summary | hierarchical
  --budget, -b TOKENS       Max context tokens per iteration (default: 8000)
  --recent-turns N           Recent turns to keep verbatim (default: 3)
  --medium-turns N           Medium turns as bullets (default: 5)
  --summary-chars N          Max chars for summary block (default: 500)

Hermes:
  --model, -m MODEL         Hermes model override
  --provider PROVIDER        Hermes provider override
  --profile, -p NAME         Hermes profile
  --workdir, -w DIR          Working directory

Output:
  --output, -o DIR           Output directory
  --export FORMAT            jsonl | markdown | both (default: both)
  --no-snapshots             Don't save per-iteration snapshots

Synthesis:
  --synthesis TEXT           Custom synthesis instruction
  --synthesis-file FILE      Read synthesis instruction from file
```

## Output

Each run creates:
- `loop_log.jsonl` — structured log of every event
- `snapshots/iter_000.json`, `iter_001.json`, ... — full conversation state at each iteration
- `conversation.jsonl` — full conversation export
- `conversation.md` — human-readable markdown export

## Architecture

```
recursive_feedback_loop/
├── __init__.py
├── cli.py              # CLI entry point (rfl command)
├── config.py           # LoopConfig dataclass
├── loop_runner.py      # Core orchestration engine
├── compaction.py       # Three compaction strategies
└── session_reader.py   # Parse Hermes JSONL sessions
```

## Extending

### Custom Compaction Strategy

```python
from recursive_feedback_loop.compaction import CompactionStrategy, register_strategy

class MyStrategy(CompactionStrategy):
    def name(self) -> str:
        return "my_strategy"

    def compact(self, conversation, token_budget: int) -> str:
        # Your logic here
        ...

register_strategy("my_strategy", MyStrategy)
```

### Programmatically

```python
from recursive_feedback_loop.config import LoopConfig
from recursive_feedback_loop.loop_runner import LoopRunner

config = LoopConfig(
    seed_prompt="Explore the concept of infinity",
    max_iterations=15,
    compaction_strategy="hierarchical",
    max_context_tokens=10000,
    hermes_model="anthropic/claude-sonnet-4",
)

runner = LoopRunner(config)
state = runner.run()

print(f"Completed {state.iteration + 1} iterations")
for turn in state.conversation.turns:
    print(f"[{turn.role}] {turn.content[:100]}...")
```

## Tests

```bash
python3 -m pytest tests/ -q
```
