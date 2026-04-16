"""Report generation for self-audit results.

Outputs:
  - Markdown: grouped by severity, file:line links, actionable
  - JSON: machine-readable for CI integration
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .issue_parser import Issue, ConvergenceResult, SEVERITY_LEVELS


def generate_report(
    issues: list[Issue],
    issue_history: list[list[Issue]],
    convergence: ConvergenceResult,
    project_path: str,
    output_format: str = "markdown",
    severity_threshold: str = "",
) -> str:
    """Generate a self-audit report.

    Args:
        issues: Final deduplicated issue list.
        issue_history: Issues per iteration (for stats).
        convergence: Convergence result.
        project_path: Path to the audited project.
        output_format: 'markdown' or 'json'.
        severity_threshold: If set, only include issues at/above this.

    Returns:
        Report as a string.
    """
    if output_format == "json":
        return _generate_json(issues, issue_history, convergence, project_path)
    return _generate_markdown(issues, issue_history, convergence, project_path, severity_threshold)


def write_report(
    issues: list[Issue],
    issue_history: list[list[Issue]],
    convergence: ConvergenceResult,
    project_path: str,
    output_path: str,
    output_format: str = "markdown",
    severity_threshold: str = "",
) -> Path:
    """Generate and write report to a file.

    Returns the path to the written file.
    """
    report = generate_report(
        issues, issue_history, convergence, project_path,
        output_format, severity_threshold,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)
    return path


def _generate_markdown(
    issues: list[Issue],
    issue_history: list[list[Issue]],
    convergence: ConvergenceResult,
    project_path: str,
    severity_threshold: str,
) -> str:
    """Generate markdown report."""
    lines = []
    lines.append(f"# Self-Audit Report: {Path(project_path).name}")
    lines.append(f"")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Project: `{project_path}`")
    lines.append(f"Iterations: {convergence.iteration_count}")
    lines.append(f"Converged: {'Yes' if convergence.converged else 'No'}")
    if severity_threshold:
        lines.append(f"Severity threshold: {severity_threshold}")
    lines.append(f"")

    # Summary header
    by_severity = _count_by_severity(issues)
    lines.append(f"## Summary")
    lines.append(f"")
    lines.append(f"| Severity | Count |")
    lines.append(f"|----------|-------|")
    for sev in SEVERITY_LEVELS:
        count = by_severity.get(sev, 0)
        if count > 0:
            lines.append(f"| {sev} | {count} |")
    lines.append(f"| **TOTAL** | **{len(issues)}** |")
    lines.append(f"")

    # Convergence stats
    lines.append(f"## Convergence")
    lines.append(f"")
    lines.append(f"- Iterations run: {convergence.iteration_count}")
    lines.append(f"- Converged: {'Yes (no new issues at threshold)' if convergence.converged else 'No'}")
    lines.append(f"- New issues in last iteration: {len(convergence.new_issues)}")
    lines.append(f"- Confirmed issues: {len(convergence.confirmed_issues)}")
    lines.append(f"")

    # Issues per iteration
    lines.append(f"### Issues per iteration")
    lines.append(f"")
    lines.append(f"| Iteration | Issues | CRITICAL | HIGH | MEDIUM | LOW |")
    lines.append(f"|-----------|--------|----------|------|--------|-----|")
    for i, iter_issues in enumerate(issue_history):
        iter_counts = _count_by_severity(iter_issues)
        lines.append(
            f"| {i} | {len(iter_issues)} | "
            f"{iter_counts.get('CRITICAL', 0)} | "
            f"{iter_counts.get('HIGH', 0)} | "
            f"{iter_counts.get('MEDIUM', 0)} | "
            f"{iter_counts.get('LOW', 0)} |"
        )
    lines.append(f"")

    # Most problematic files
    file_counts = {}
    for issue in issues:
        file_counts[issue.file] = file_counts.get(issue.file, 0) + 1
    hot_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    if hot_files:
        lines.append(f"### Most Problematic Files")
        lines.append(f"")
        for f, count in hot_files:
            lines.append(f"- `{f}` -- {count} issues")
        lines.append(f"")

    # Detailed issues by severity
    lines.append(f"## Issues")
    lines.append(f"")
    for sev in SEVERITY_LEVELS:
        sev_issues = [i for i in issues if i.severity.upper() == sev]
        if not sev_issues:
            continue

        lines.append(f"### {sev} ({len(sev_issues)})")
        lines.append(f"")

        for idx, issue in enumerate(sev_issues, 1):
            location = f"`{issue.file}`"
            if issue.line:
                location += f":{issue.line}"
            lines.append(f"#### {idx}. {location}")
            lines.append(f"")
            lines.append(f"{issue.description}")
            lines.append(f"")
            if issue.impact:
                lines.append(f"**Impact:** {issue.impact}")
                lines.append(f"")
            if issue.fix:
                lines.append(f"**Fix:** {issue.fix}")
                lines.append(f"")
            lines.append(f"Confidence: {issue.confidence:.0%} | Found in iteration {issue.iteration}")
            lines.append(f"")
            lines.append(f"---")
            lines.append(f"")

    return "\n".join(lines)


def _generate_json(
    issues: list[Issue],
    issue_history: list[list[Issue]],
    convergence: ConvergenceResult,
    project_path: str,
) -> str:
    """Generate machine-readable JSON report."""
    report = {
        "generated": datetime.now().isoformat(),
        "project": str(project_path),
        "convergence": convergence.to_dict(),
        "total_issues": len(issues),
        "by_severity": _count_by_severity(issues),
        "iterations": [
            {
                "iteration": i,
                "issue_count": len(iter_issues),
                "issues": [issue.to_dict() for issue in iter_issues],
            }
            for i, iter_issues in enumerate(issue_history)
        ],
        "issues": [issue.to_dict() for issue in issues],
    }
    return json.dumps(report, indent=2)


def _count_by_severity(issues: list[Issue]) -> dict:
    """Count issues by severity level."""
    counts = {}
    for issue in issues:
        sev = issue.severity.upper()
        counts[sev] = counts.get(sev, 0) + 1
    return counts
