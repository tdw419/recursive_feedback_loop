"""Parse structured issues from LLM output and detect convergence.

Three-stage parsing pipeline:
  Stage 1 (regex): [SEVERITY] file:line -- description
  Stage 2 (structural): numbered lists, markdown tables, bullet lists
  Stage 3 (LLM fallback): ask the LLM to extract structured issues

Convergence = no new CRITICAL or HIGH issues found in latest iteration.
"""

import re
import json
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path


# Severity levels in priority order
SEVERITY_LEVELS = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
SEVERITY_ORDER = {s: i for i, s in enumerate(SEVERITY_LEVELS)}


@dataclass
class Issue:
    """A single parsed issue from an audit iteration."""
    severity: str          # CRITICAL, HIGH, MEDIUM, LOW
    file: str              # path/to/file.py
    line: Optional[int]    # line number (None if not specified)
    description: str       # what's wrong
    impact: str = ""       # why it matters
    fix: str = ""          # concrete fix
    confidence: float = 0.8  # 0-1, how confident we are in this parse
    iteration: int = 0     # which iteration found this
    raw_text: str = ""     # original text segment

    def severity_rank(self) -> int:
        """Lower is more severe."""
        return SEVERITY_ORDER.get(self.severity.upper(), 99)

    def is_high_or_above(self) -> bool:
        return self.severity_rank() <= SEVERITY_ORDER.get("HIGH", 1)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw_text", None)
        return d

    def fingerprint(self) -> str:
        """Fuzzy identity for dedup. Normalizes file + first 60 chars of description."""
        desc = self.description.lower().strip()[:60]
        file_norm = self.file.replace("\\", "/")
        # Normalize: strip leading ./ prefix for dedup
        if file_norm.startswith("./"):
            file_norm = file_norm[2:]
        return f"{file_norm}:{desc}"


@dataclass
class ConvergenceResult:
    """Result of checking convergence across iterations."""
    converged: bool
    iteration_count: int
    new_issues: list = field(default_factory=list)
    confirmed_issues: list = field(default_factory=list)
    refuted_issues: list = field(default_factory=list)
    total_by_severity: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "converged": self.converged,
            "iteration_count": self.iteration_count,
            "new_issues": [i.to_dict() for i in self.new_issues],
            "confirmed_issues": [i.to_dict() for i in self.confirmed_issues],
            "refuted_issues": [i.to_dict() for i in self.refuted_issues],
            "total_by_severity": self.total_by_severity,
        }


def parse_issues(llm_output: str, iteration: int = 0) -> list[Issue]:
    """Parse structured issues from LLM output.

    Tries three stages in order:
      1. Regex: [SEVERITY] file:line -- description
      2. Structural: numbered lists with severity keywords
      3. Returns best-effort if stages 1-2 produce nothing
    """
    if not llm_output or not llm_output.strip():
        return []

    issues = _parse_regex(llm_output, iteration)
    if issues:
        return issues

    issues = _parse_structural(llm_output, iteration)
    if issues:
        return issues

    # Stage 3: best-effort extraction from any severity mentions
    return _parse_best_effort(llm_output, iteration)


def _parse_regex(text: str, iteration: int) -> list[Issue]:
    """Stage 1: Extract issues matching [SEVERITY] file:line -- description."""
    issues = []
    seen = set()

    # Pattern: [SEVERITY] path/to/file:line -- description
    # Handles: [CRITICAL], [HIGH], [MEDIUM], [LOW] with file:line refs
    pattern = re.compile(
        r'\[(CRITICAL|HIGH|MEDIUM|LOW)\]\s*'
        r'([\w./\\-]+(?:/[\w./\\-]+)*)'       # file path (no colons)
        r'(?::(\d+)(?:-\d+)?)?\s*'            # optional :line or :line-line
        r'(?:--|[-\u2013\u2014])\s*'            # -- or dash separator
        r'(.+?)(?=\n\s*\[|\n\s*\d+\.|\n\n|$)', # description until next issue or end
        re.IGNORECASE | re.DOTALL,
    )

    for m in pattern.finditer(text):
        severity = m.group(1).upper()
        file_path = m.group(2).strip()
        line_num = int(m.group(3)) if m.group(3) else None
        description = (m.group(4) or "").strip()

        # Clean up description
        description = _clean_description(description)
        if not description or len(description) < 5:
            continue

        issue = Issue(
            severity=severity,
            file=_clean_file_path(file_path),
            line=line_num,
            description=description,
            confidence=0.9,
            iteration=iteration,
            raw_text=m.group(0)[:200],
        )

        fp = issue.fingerprint()
        if fp not in seen:
            seen.add(fp)
            issues.append(issue)

    # Also try: simpler pattern without file:line
    # SEVERITY: description (file:line)
    if not issues:
        simple_pattern = re.compile(
            r'(?:^|\n)\s*(?:\d+\.\s*)?(CRITICAL|HIGH|MEDIUM|LOW)[\s:]+(.+?)(?:\((.+?:\d+)\))?',
            re.IGNORECASE,
        )
        for m in simple_pattern.finditer(text):
            severity = m.group(1).upper()
            description = m.group(2).strip()
            file_ref = m.group(3) or ""

            file_path, line_num = _parse_file_ref(file_ref)
            description = _clean_description(description)
            if not description or len(description) < 5:
                continue

            issue = Issue(
                severity=severity,
                file=file_path,
                line=line_num,
                description=description,
                confidence=0.7,
                iteration=iteration,
            )

            fp = issue.fingerprint()
            if fp not in seen:
                seen.add(fp)
                issues.append(issue)

    return issues


def _parse_structural(text: str, iteration: int) -> list[Issue]:
    """Stage 2: Parse numbered lists or bullet lists with severity keywords."""
    issues = []
    seen = set()

    # Match numbered items: "1. **[CRITICAL]** ..." or "1. CRITICAL: ..."
    # or bullet items: "- CRITICAL: ..." or "* CRITICAL: ..."
    item_pattern = re.compile(
        r'(?:^|\n)\s*(?:\d+\.\s*|[-*]\s*)'        # list marker
        r'(?:\*\*)?'                                 # optional bold
        r'\[?(CRITICAL|HIGH|MEDIUM|LOW)\]?'         # severity
        r'(?:\*\*)?'                                 # optional bold close
        r'[\s:]+\s*'                                 # separator
        r'(.+?)(?=\n\s*(?:\d+\.|[-*])|\n\n|$)',     # description
        re.IGNORECASE | re.DOTALL,
    )

    for m in item_pattern.finditer(text):
        severity = m.group(1).upper()
        description = m.group(2).strip()
        description = _clean_description(description)
        if not description or len(description) < 10:
            continue

        # Try to extract file:line from the description
        file_path, line_num = _extract_file_ref_from_text(description)
        if file_path:
            # Remove the file:line from description if present at start
            description = re.sub(
                r'^[`*]?[\w./\\-]+:\d+[`*]?\s*[-\u2013\u2014]\s*',
                '', description
            ).strip()

        issue = Issue(
            severity=severity,
            file=file_path,
            line=line_num,
            description=description[:300],
            confidence=0.6,
            iteration=iteration,
        )

        fp = issue.fingerprint()
        if fp not in seen:
            seen.add(fp)
            issues.append(issue)

    return issues


def _parse_best_effort(text: str, iteration: int) -> list[Issue]:
    """Stage 3: Find any severity mentions and extract what we can."""
    issues = []
    seen = set()

    # Find all severity mentions with surrounding context
    for sev in SEVERITY_LEVELS:
        for m in re.finditer(
            rf'(?:^|\n)(.*?{sev}.*?)(?=\n|$)',
            text, re.IGNORECASE,
        ):
            line = m.group(1).strip()
            if len(line) < 10:
                continue

            # Try to extract file ref
            file_path, line_num = _extract_file_ref_from_text(line)

            # Clean the line
            desc = re.sub(
                rf'\b{sev}\b[\s:]*', '', line, count=1, flags=re.IGNORECASE
            ).strip().strip('-*:.')
            desc = desc[:300]

            if not desc:
                continue

            issue = Issue(
                severity=sev,
                file=file_path,
                line=line_num,
                description=desc,
                confidence=0.3,
                iteration=iteration,
            )

            fp = issue.fingerprint()
            if fp not in seen:
                seen.add(fp)
                issues.append(issue)

    return issues


def _clean_description(text: str) -> str:
    """Clean up a description string, extracting Impact/Fix if present."""
    # Remove markdown formatting
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Truncate if Impact/Fix got merged in
    # Try to cut at "Impact:" or "Fix:" boundaries
    for marker in [' Impact:', ' Fix:']:
        idx = text.find(marker)
        if idx > 20:
            text = text[:idx].strip()
            break
    return text[:500]


def _clean_file_path(path: str) -> str:
    """Clean up a file path."""
    path = path.strip().strip('`*"\'>')
    path = path.replace('\\', '/')
    # Remove trailing punctuation that got captured
    path = re.sub(r'[:\s]+$', '', path)
    return path


def _parse_file_ref(ref: str) -> tuple[str, Optional[int]]:
    """Parse a file:line reference like 'path/to/file.py:42'."""
    if not ref:
        return "", None
    ref = ref.strip().strip('`*"\'>')
    m = re.match(r'(.+?):(\d+)', ref)
    if m:
        return _clean_file_path(m.group(1)), int(m.group(2))
    return _clean_file_path(ref), None


def _extract_file_ref_from_text(text: str) -> tuple[str, Optional[int]]:
    """Try to extract a file:line reference from free-form text."""
    # Match file.py:42 or file.py:42-50 or ./path/file.py:42
    m = re.search(r'([`*]?)((?:\.{0,2}/)?[\w./\\-]+\.\w+)(?::(\d+)(?:-\d+)?)\1', text)
    if m:
        return _clean_file_path(m.group(2)), int(m.group(3)) if m.group(3) else None

    # Match just filename.ext without line number
    m = re.search(r'([`*]?)((?:\.{0,2}/)?[\w./\\-]+\.\w+)\1', text)
    if m:
        path = _clean_file_path(m.group(2))
        # Skip common non-file patterns
        if not path.startswith(('http', 'www', 'Example')):
            return path, None

    return "", None


def check_convergence(
    issue_history: list[list[Issue]],
    severity_threshold: str = "HIGH",
) -> ConvergenceResult:
    """Check if the loop has converged.

    Convergence = no new issues at or above severity_threshold found
    in the latest iteration that weren't in previous iterations.

    Args:
        issue_history: List of issue lists, one per iteration.
        severity_threshold: Minimum severity to consider for convergence.
            Issues below this threshold don't affect convergence.

    Returns:
        ConvergenceResult with convergence status and issue tracking.
    """
    if not issue_history:
        return ConvergenceResult(
            converged=True,
            iteration_count=0,
            total_by_severity={},
        )

    threshold_rank = SEVERITY_ORDER.get(severity_threshold.upper(), 1)

    # Build fingerprints of all previously seen issues
    all_previous_fps = set()
    for iter_issues in issue_history[:-1]:
        for issue in iter_issues:
            if issue.severity_rank() <= threshold_rank:
                all_previous_fps.add(issue.fingerprint())

    # Analyze latest iteration
    latest = issue_history[-1]
    new_issues = []
    confirmed_issues = []

    for issue in latest:
        if issue.severity_rank() > threshold_rank:
            continue  # below threshold, doesn't affect convergence

        fp = issue.fingerprint()
        if fp in all_previous_fps:
            confirmed_issues.append(issue)
        else:
            new_issues.append(issue)

    # Count total by severity across all iterations
    total_by_severity = {}
    all_issues = [i for iters in issue_history for i in iters]
    # Deduplicate by fingerprint
    seen_fps = set()
    unique_issues = []
    for issue in all_issues:
        fp = issue.fingerprint()
        if fp not in seen_fps:
            seen_fps.add(fp)
            unique_issues.append(issue)

    for issue in unique_issues:
        sev = issue.severity.upper()
        total_by_severity[sev] = total_by_severity.get(sev, 0) + 1

    # Check for refuted issues: issues from previous iterations that
    # the latest iteration explicitly denies (heuristic: look for
    # "refuted", "false positive", "not a bug" patterns)
    refuted_issues = []

    converged = len(new_issues) == 0

    return ConvergenceResult(
        converged=converged,
        iteration_count=len(issue_history),
        new_issues=new_issues,
        confirmed_issues=confirmed_issues,
        refuted_issues=refuted_issues,
        total_by_severity=total_by_severity,
    )


def deduplicate_issues(issues: list[Issue]) -> list[Issue]:
    """Deduplicate issues by fingerprint, keeping highest confidence."""
    best = {}
    for issue in issues:
        fp = issue.fingerprint()
        if fp not in best or issue.confidence > best[fp].confidence:
            best[fp] = issue

    return sorted(best.values(), key=lambda i: (i.severity_rank(), i.file))


def filter_by_severity(issues: list[Issue], threshold: str) -> list[Issue]:
    """Filter issues to only include those at or above threshold."""
    threshold_rank = SEVERITY_ORDER.get(threshold.upper(), 1)
    return [i for i in issues if i.severity_rank() <= threshold_rank]
