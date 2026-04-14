# RFL Autoresearch — Agent Instructions

You are an autonomous researcher tuning the Recursive Feedback Loop (RFL) tool.

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr14`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `recursive_feedback_loop/config.py` — all tunable parameters
   - `recursive_feedback_loop/compaction.py` — compaction strategies (Hierarchical, SlidingWindow, RollingSummary)
   - `recursive_feedback_loop/loop_runner.py` — core loop, deduper, prompt synthesis
   - `recursive_feedback_loop/session_mode.py` — tmux session mode
   - `recursive_feedback_loop/session_reader.py` — session file parsing
   - `rfl_autoresearch/experiment.py` — evaluation harness (DO NOT MODIFY)
   - `rfl_autoresearch/README.md` — overview of the experiment framework
4. **Verify setup**: run `uv run python -m rfl_autoresearch.experiment -n 2 -d baseline` to establish a baseline.
5. **Initialize results.tsv**: Create `rfl_autoresearch/results.tsv` with just the header if it doesn't exist.

Once you confirm setup, begin experimentation.

## Experimentation

Each experiment runs for a **fixed iteration budget** (default: 3 iterations for speed). The evaluation harness measures:
- **delta_ratio** (0-1): How much new content each iteration adds vs repeating. Target: >0.6
- **specificity_score**: Concrete references per 1K words. Target: >5
- **depth_progression** (0-1): Does output deepen across iterations. Target: >0.5
- **composite_score**: Weighted combination. **This is the primary metric to optimize.**

You launch experiments as:
```bash
uv run python -m rfl_autoresearch.experiment -n 3 -d "description of what you changed" > run.log 2>&1
```

Then read results:
```bash
grep -E "^(composite_score|delta_ratio|specificity|depth|status):" run.log
```

**What you CAN do:**
- Modify any file in `recursive_feedback_loop/` — config, compaction, loop runner, session mode
- Add new compaction strategies
- Rewrite prompt templates
- Change deduper logic
- Adjust hardcoded thresholds
- Add new CLI parameters

**What you CANNOT do:**
- Modify `rfl_autoresearch/experiment.py` — it is the evaluation harness (read-only)
- Modify the scoring functions in the experiment harness
- Install new packages (only use what's already available)

## The metric

**The goal is simple: get the highest composite_score.** The scoring is:
- 35% weight on delta_ratio (low repetition)
- 30% weight on specificity (concrete details)
- 20% weight on depth progression (deepening across iterations)
- 15% weight on dedup efficiency (catching duplicate output)

**The first run**: Always establish the baseline with the default config.

## Logging results

When an experiment finishes, record it in `rfl_autoresearch/results.tsv`. The experiment.py auto-logs, but verify the entry is there.

## The experiment loop

Run on a dedicated branch: `autoresearch/<tag>`.

LOOP FOREVER:

1. Look at git state: current branch/commit
2. Modify RFL code with an experimental idea
3. git commit
4. Run the experiment: `uv run python -m rfl_autoresearch.experiment -n 3 -d "description" > run.log 2>&1`
5. Read results: `grep -E "^(composite_score|status):" run.log`
6. If grep output is empty, the run crashed. Check `tail -50 run.log` for the error.
7. If composite_score improved (higher), "advance" the branch by keeping the commit
8. If composite_score is equal or worse, `git reset --hard HEAD~1` to revert
9. If the run crashes, fix the bug and re-run. If unfixable after 3 tries, skip it.

**Timeout**: Each experiment should take ~5-10 minutes (3 iterations × 2-3 min each + evaluation). If a run exceeds 20 minutes, kill it and treat as failure.

**Crashes**: Fix simple bugs (typos, import errors). Skip fundamentally broken ideas.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. You are autonomous. If you run out of ideas, think harder — re-read the code for new angles, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you.

## What to experiment with

### High-impact areas (start here)
1. **synthesis_instruction prompt** in config.py — this is the "go deeper" instruction that drives each iteration. Small changes here have outsized effects.
2. **Deduper thresholds** in loop_runner.py — the chunk sizes [800, 500, 300, 200, 100] and overlap detection
3. **Hierarchical compaction parameters** — recent_turns, medium_turns, summary_max_chars

### Medium-impact areas
4. **_turn_to_bullets** in compaction.py — the sentence extraction for medium-tier context
5. **_build_synthesis_prompt** in loop_runner.py — how context is framed for the next iteration
6. **Token estimation** — chars/4 is crude. Could use a better heuristic.

### Structural experiments (bigger changes)
7. **New compaction strategy** — e.g., "keypoints" that extracts numbered findings per turn
8. **Adaptive budget** — increase token budget in later iterations when specificity is high
9. **Multi-model compaction** — use a cheaper model for summarization, better model for synthesis

### Simplicity criterion
All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing code and getting equal or better results is a great outcome.
