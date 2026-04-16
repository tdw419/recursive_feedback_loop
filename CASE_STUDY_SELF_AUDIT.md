# RFL Self-Audit Case Study

How the RFL found 8 bugs in its own source code, and what it teaches about recursive feedback.

## The Setup

The RFL (~2,000 LOC Python) was pointed at itself with a seed prompt describing its own architecture and two known weakness areas. The goal: find real bugs that no human was looking for.

**Command:**
```bash
rfl run --from-file /tmp/rfl_seed_self.md -n 4 -b 10000 -s hierarchical \
  -w ~/zion/projects/recursive_feedback_loop -o rfl_output/self_audit -t 900
```

The seed prompt listed all source files, their responsibilities, two proposed improvement areas (multi-model phases, smarter compaction), and asked the AI to read actual source code and find exact failure modes.

## The Loop: What Happened Each Iteration

### Iteration 0: Surface Scan (7 issues)

Hermes read all 6 source files (cli.py, config.py, compaction.py, loop_runner.py, session_reader.py, session_mode.py). It found the obvious structural issues -- the O(n^2) re-summarization in compaction.py, the synthesis wrapper overhead, the bullet mangling. These were good catches but none were critical correctness bugs.

### Iteration 1: Verification and Depth (16 issues)

This iteration corrected mistakes from iteration 0 (some cited line numbers were wrong) and went deeper into loop_runner.py and compaction.py. It found the `_clean_hermes_output` fallback bug and the deduper threshold gap. But it still hadn't found the worst bug.

### Iteration 2: The Kill Shot (19+ issues)

This iteration found **the most damaging bug in the entire codebase**: `cli.py:145` -- `args.synthesis or ""`. Every single CLI invocation without an explicit `--synthesis` flag was silently replacing the carefully crafted 5-rule depth prompt with an empty string. The RFL had been running iterations 1-N with no synthesis instruction at all, relying entirely on the iteration pattern itself for depth.

This is the key insight: **iteration 2 found the worst bug because iterations 0 and 1 exhausted all the obvious targets.** The model had to look at secondary code paths, CLI argument parsing, config resolution -- the code nobody looks at first.

### Iteration 3: Cross-File Verification (22 issues)

This iteration cross-referenced findings from all previous iterations. It verified the CLI synthesis bug against the actual config.py default. It found the session_reader.py UnboundLocalError edge case. It critiqued its own scoring harness.

### Iteration 4: Final Verification (22 confirmed issues)

The last iteration verified every claim from iterations 0-3 against the actual source code, checking loop_log.jsonl data to confirm compaction overhead numbers. It discarded false positives and produced a final prioritized list.

## The Bugs Found (All Fixed in commit a0270bf)

### CRITICAL (3)

| # | File | Bug | Impact | Fix |
|---|------|-----|--------|-----|
| 1 | cli.py:145 | `args.synthesis or ""` overwrites default synthesis instruction | Every CLI run without `--synthesis` ran with NO depth escalation | `args.synthesis if args.synthesis else None` |
| 2 | loop_runner.py:474 | `_clean_hermes_output` returns raw on empty clean | UI artifacts (box-drawing chars) leak into conversation history | Return empty string |
| 3 | session_reader.py:103 | UnboundLocalError on empty file | Crash when session file is empty | Track `last_i` separately |

### HIGH (5)

| # | File | Bug | Impact | Fix |
|---|------|-----|--------|-----|
| 4 | loop_runner.py:482 | Synthesis wrapper adds ~200 tokens/iter | Wastes 1000+ tokens over 5 iterations | Trimmed wrapper text |
| 5 | compaction.py:75 | RollingSummary re-summarizes ALL old turns | O(n^2) -- each iteration re-processes everything | Track `_last_summarized_idx` |
| 6 | compaction.py:184 | Hierarchical re-summarizes unchanged old tier | Same O(n^2) waste as RollingSummary | Track `_last_old_idx` |
| 7 | compaction.py:274 | `_turn_to_bullets` replaces `!` and `?` with `.` | Destroys `!=`, version strings, file extensions | Regex sentence-boundary split |
| 8 | loop_runner.py:36 | Deduper skips responses < 400 chars | Short Hermes responses always duplicated | Lower threshold to 100 |

## Why This Worked: The Three Structural Reasons

### 1. Mode-Forcing Through Iteration

A single LLM pass has one mode -- it scans, or verifies, or synthesizes. The RFL mechanically forces different modes per iteration:

- Iteration 0 scans: broad, fast, claims without proof
- Iteration 1 verifies: reads actual files, corrects mistakes
- Iteration 2 goes deeper: secondary targets, code paths nobody looks at first
- Iteration 3 cross-cuts: spans multiple files, checks for cascading effects
- Iteration 4 verifies: confirms or denies every prior claim

The CLI synthesis bug (the worst one) was found in iteration 2 because iterations 0-1 had already exhausted all the interesting algorithmic targets. The model was forced to look at argument parsing and config resolution -- the boring glue code where the bug lived.

### 2. LLMs Are Better Critics Than Creators

A model generating a fresh analysis will hallucinate. The same model reading its own previous analysis and asked "what's wrong with this?" catches errors. The RFL exploits this asymmetry -- each iteration critiques the previous one. Iteration 1 corrected iteration 0's wrong line numbers. Iteration 3 caught false positives from iteration 2.

### 3. Compaction Prevents Lazy Repetition

By stripping detail between iterations, the model must reconstruct from partial information. It can't parrot its prior answer. The deduper enforces this mechanically. The synthesis prompt enforces it linguistically. Together they prevent the model from taking the lazy path of restating what it already said.

## The Crucial Insight: Bug #1 Changed Everything

The most important finding was also the most ironic. The synthesis instruction overwrite (bug #1) meant that every previous RFL run -- including the Geometry OS audit that found 5 real bugs, and the infinite map feature build -- was running without any synthesis instruction at all. The iteration pattern alone was powerful enough to find real bugs.

With the fix in place, every new RFL run now gets the structured 5-rule depth prompt:
1. Zero repetition
2. Provide EXACT fixes or ROOT CAUSES
3. Escalate specificity (line numbers, function names)
4. Validate proposed fixes against edge cases
5. At least 3 NEW observations per iteration

This should be a measurable quality improvement. The autoresearch already showed the synthesis prompt is the #1 lever (composite score jumped from 0.504 to 0.538).

## The Compaction Overhead Bug

The self-audit also confirmed the compaction overhead bug through live data:

| Iteration | Original tokens | Compacted tokens | Change |
|-----------|----------------|-----------------|--------|
| 1 | 6,582 | 6,601 | +0.3% (EXPANSION) |
| 2 | 13,862 | 11,997 | -13.5% |
| 3 | 21,689 | 11,997 | -44.7% |
| 4 | 26,546 | 11,997 | -54.8% |

Iteration 1's compaction actually EXPANDED the context. The wrapper text and formatting headers exceeded the original content size. By iteration 3-4, over half the context was lost to compression. The fixes to incremental summarization (tracking what's already been processed) should eliminate the O(n^2) waste and reduce the overhead.

## What We Learned About the RFL Process

1. **The last iteration is often the most valuable.** If we'd stopped at 3, we'd have missed the session_reader edge case and the final cross-verification. Optimal iterations = "one more than you think you need."

2. **The seed prompt matters more than model quality.** The seed prompt listed specific files, known weaknesses, and asked for exact code citations. That's what made the output actionable.

3. **Self-referential audits are uniquely effective.** The RFL understanding its own architecture meant each iteration could reason about failure modes more precisely than it could for unfamiliar code.

4. **Verify everything.** The RFL cited line numbers that were sometimes off by 10-20 lines. Every finding was verified against the actual source before fixing.

## Reproducing This Pattern

To run a self-audit on any tool:

```bash
# 1. Write a seed that describes the tool's own architecture
cat > /tmp/self_audit_seed.md << 'EOF'
Audit [TOOL NAME] at [PATH]

This is [TOOL] analyzing itself. Key files:
- file1.py -- what it does
- file2.py -- what it does

KNOWN WEAKNESSES TO INVESTIGATE:
- [List 4-6 known problem areas]

For each issue: 1) specific file/function/lines, 2) why it matters, 3) concrete fix.
Read the actual source. Don't guess -- cite real code.
EOF

# 2. Run the self-audit
rfl run --from-file /tmp/self_audit_seed.md -n 5 -b 12000 \
  -w /path/to/tool -o ./self_audit -t 900

# 3. Read the output
cat ./self_audit/conversation.md
```

The key ingredients:
- **Architecture description in the seed**: So the AI knows what to look at
- **Known weaknesses**: Seeds investigation into real problem areas
- **Demand for exact code citations**: Prevents hallucination
- **5 iterations**: Enough for scan → verify → deep → cross-cut → verify
- **12K budget**: Sufficient for code-heavy analysis without losing too much to compaction
