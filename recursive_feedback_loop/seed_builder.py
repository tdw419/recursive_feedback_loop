"""Auto-construct audit seed prompts from project source code.

Two modes:
  - full:  gather all source files, build comprehensive seed
  - diff:  extract git diff + dependency-traced surrounding code

The seed prompt is the most important part of self-audit. A bad seed
produces vague output. A good seed forces the LLM to cite specific
file:line references with severity ratings.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional

# File extensions we consider source code (not data, config, etc.)
SOURCE_EXTENSIONS = {
    ".py", ".pyi", ".pyx",
    ".rs", ".toml",
    ".js", ".ts", ".jsx", ".tsx", ".mjs",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx",
    ".go",
    ".rb",
    ".java", ".kt", ".kts",
    ".sh", ".bash",
    ".zig",
    ".lua",
    ".swift",
    ".rkt",
}

# Directories to skip
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".tox", "dist", "build",
    ".eggs", "*.egg-info", ".sass-cache", ".next", ".nuxt",
    "target", "vendor", "Cargo/target",
}


def build_audit_seed(
    project_path: str | Path,
    mode: str = "diff",
    extra_context: str = "",
) -> str:
    """Build an audit seed prompt for the project at project_path.

    Args:
        project_path: Root directory of the project to audit.
        mode: 'diff' (uncommitted changes only) or 'full' (entire codebase).
        extra_context: Optional additional instructions to append to seed.

    Returns:
        A seed prompt string ready for the RFL loop.
    """
    project_path = Path(project_path).resolve()

    if not project_path.is_dir():
        raise ValueError(f"Not a directory: {project_path}")

    if mode == "diff":
        return _build_diff_seed(project_path, extra_context)
    else:
        return _build_full_seed(project_path, extra_context)


def detect_mode(project_path: str | Path) -> str:
    """Auto-detect audit mode. Returns 'diff' if in a git repo with
    uncommitted changes, otherwise 'full'.
    """
    project_path = Path(project_path).resolve()

    if not _is_git_repo(project_path):
        return "full"

    diff = _get_diff(project_path)
    if diff.strip():
        return "diff"

    return "full"


def _is_git_repo(path: Path) -> bool:
    """Check if path is inside a git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, timeout=5,
            cwd=str(path),
        )
        return result.returncode == 0 and "true" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _get_diff(path: Path, cached: bool = False) -> str:
    """Get git diff output. If cached, get staged changes."""
    cmd = ["git", "diff"]
    if cached:
        cmd.append("--cached")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10,
            cwd=str(path),
        )
        return result.stdout if result.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def _get_changed_files(path: Path) -> list[str]:
    """Get list of files changed in working tree + staged."""
    files = set()
    for cmd in [
        ["git", "diff", "--name-only"],
        ["git", "diff", "--cached", "--name-only"],
        ["git", "diff", "--name-only", "HEAD"],  # all uncommitted vs HEAD
    ]:
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10,
                cwd=str(path),
            )
            if result.returncode == 0:
                for f in result.stdout.strip().splitlines():
                    if f.strip():
                        files.add(f.strip())
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    return sorted(files)


def _trace_symbol_dependencies(path: Path, changed_files: list[str]) -> dict:
    """Simple symbol-level dependency tracing.

    For each changed file, find function/class definitions, then grep
    for callers across the project. Returns {file: [caller_files]}.
    """
    dependencies = {}
    symbols_by_file = {}

    # Extract symbols from changed files
    for rel_file in changed_files:
        full_path = path / rel_file
        if not full_path.exists() or not full_path.is_file():
            continue
        try:
            content = full_path.read_text(errors="replace")
        except OSError:
            continue

        symbols = _extract_symbols(content, full_path.suffix)
        if symbols:
            symbols_by_file[rel_file] = symbols
            dependencies[rel_file] = set()

    # Find callers of those symbols across all source files
    all_symbols = set()
    for syms in symbols_by_file.values():
        all_symbols.update(syms)

    if not all_symbols:
        return {f: [] for f in changed_files}

    for source_file in _find_source_files(path):
        if str(source_file.relative_to(path)) in changed_files:
            continue
        try:
            content = source_file.read_text(errors="replace")
        except OSError:
            continue

        for symbol in all_symbols:
            if symbol in content:
                rel = str(source_file.relative_to(path))
                for cf in symbols_by_file:
                    if symbol in symbols_by_file[cf]:
                        dependencies.setdefault(cf, set()).add(rel)
                        break

    return {f: sorted(deps) for f, deps in dependencies.items()}


def _extract_symbols(content: str, suffix: str) -> list[str]:
    """Extract function/class/method names from source code."""
    import re
    symbols = []

    if suffix in (".py", ".pyi", ".pyx"):
        # Python: def foo, class Foo, async def foo
        for m in re.finditer(r'^(?:async\s+)?(?:def|class)\s+(\w+)', content, re.MULTILINE):
            symbols.append(m.group(1))
    elif suffix in (".js", ".ts", ".jsx", ".tsx", ".mjs"):
        # JS/TS: function foo, const foo =, class Foo, export function
        for m in re.finditer(
            r'(?:export\s+)?(?:async\s+)?(?:function|class)\s+(\w+)|'
            r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:function|\()',
            content
        ):
            name = m.group(1) or m.group(2)
            if name:
                symbols.append(name)
    elif suffix in (".rs",):
        # Rust: fn foo, struct Foo, enum Foo, trait Foo, impl Foo
        for m in re.finditer(r'(?:pub\s+)?(?:async\s+)?(?:fn|struct|enum|trait)\s+(\w+)', content):
            symbols.append(m.group(1))
    elif suffix in (".go",):
        # Go: func Foo, type Foo struct, type Foo interface
        for m in re.finditer(r'func\s+(?:\([^)]+\)\s+)?(\w+)|type\s+(\w+)\s+(?:struct|interface)', content):
            symbols.append(m.group(1) or m.group(2))

    return symbols[:50]  # cap at 50 symbols per file


def _find_source_files(path: Path) -> list[Path]:
    """Find all source files in a project, respecting SKIP_DIRS."""
    files = []
    for root, dirs, filenames in os.walk(path):
        # Skip hidden and excluded dirs in-place
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for fn in filenames:
            if Path(fn).suffix in SOURCE_EXTENSIONS:
                files.append(Path(root) / fn)
    return files


def _gather_source_inventory(path: Path, max_files: int = 80) -> list[dict]:
    """Gather a file inventory: [{path, lines, lang}]."""
    files = _find_source_files(path)[:max_files]
    inventory = []
    for f in files:
        try:
            lines = sum(1 for _ in open(f, errors="replace"))
        except OSError:
            lines = 0
        rel = str(f.relative_to(path))
        lang = f.suffix.lstrip(".")
        inventory.append({"path": rel, "lines": lines, "lang": lang})
    return sorted(inventory, key=lambda x: x["lines"], reverse=True)


def _read_file_header(path: Path, max_lines: int = 30) -> str:
    """Read the first N lines of a file (imports, signatures, docstrings)."""
    try:
        lines = []
        with open(path, errors="replace") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                lines.append(line.rstrip())
        return "\n".join(lines)
    except OSError:
        return ""


def _build_full_seed(project_path: Path, extra_context: str) -> str:
    """Build seed for full codebase audit."""
    inventory = _gather_source_inventory(project_path)
    if not inventory:
        return _build_minimal_seed(project_path, extra_context)

    # Build file listing
    file_listing = []
    total_lines = 0
    for entry in inventory[:60]:
        file_listing.append(f"  {entry['path']}  ({entry['lines']} lines, {entry['lang']})")
        total_lines += entry["lines"]

    if len(inventory) > 60:
        remaining = len(inventory) - 60
        file_listing.append(f"  ... and {remaining} more files")

    # Include top file headers for the largest files
    headers = []
    for entry in inventory[:8]:
        full_path = project_path / entry["path"]
        header = _read_file_header(full_path, 20)
        if header:
            headers.append(f"--- {entry['path']} (first 20 lines) ---\n{header}")

    header_block = "\n\n".join(headers) if headers else ""

    seed = f"""SELF-AUDIT: Recursive audit of codebase at {project_path.name}

You are performing a deep self-audit of this project. Your job is to find
REAL bugs, design flaws, and correctness issues -- not style nits.

PROJECT: {project_path.name}
FILES: {len(inventory)} source files, ~{total_lines} total lines

FILE INVENTORY (sorted by size):
{chr(10).join(file_listing)}
{f"\n\nFILE HEADERS:\n{header_block}" if header_block else ""}
MODE-FORCING RULES (you MUST follow these):
1. Every issue MUST have: [SEVERITY] file:line -- description
2. Severity is one of: CRITICAL, HIGH, MEDIUM, LOW
3. You MUST cite actual file names and line numbers from the inventory above
4. NO vague observations like "there might be an issue with error handling"
5. Each issue must include: exact location, what's wrong, WHY it matters, concrete fix
6. Read the actual source files using your tools. Do NOT guess.

YOUR APPROACH:
- Iteration 0: Scan all files. List every potential issue with exact locations.
- Iteration 1+: Verify iteration 0's claims against the actual code. Correct mistakes.
  Find issues the previous iteration missed. Go deeper into secondary code paths.
- Each iteration MUST find at least 3 NEW issues or verify/refute previous claims.

OUTPUT FORMAT (required):
For each issue:
[SEVERITY] path/to/file:line_number -- Description of the issue
  Impact: Why this matters
  Fix: Concrete fix or approach

End each iteration with a summary:
TOTAL: X issues (Y CRITICAL, Z HIGH, W MEDIUM, V LOW)
NEW THIS ITERATION: N issues
{f"\nADDITIONAL CONTEXT:\n{extra_context}" if extra_context else ""}"""

    return seed


def _build_diff_seed(project_path: Path, extra_context: str) -> str:
    """Build seed focused on uncommitted changes."""
    # Get the actual diff content
    diff_working = _get_diff(project_path, cached=False)
    diff_staged = _get_diff(project_path, cached=True)
    changed_files = _get_changed_files(project_path)
    dependencies = _trace_symbol_dependencies(project_path, changed_files)

    if not changed_files:
        return _build_full_seed(project_path, extra_context)

    # Combine diffs
    diff_parts = []
    if diff_staged.strip():
        diff_parts.append("--- STAGED CHANGES ---\n" + diff_staged)
    if diff_working.strip():
        diff_parts.append("--- UNSTAGED CHANGES ---\n" + diff_working)
    diff_content = "\n\n".join(diff_parts)

    # Truncate diff if huge
    if len(diff_content) > 15000:
        diff_content = diff_content[:15000] + "\n\n[... diff truncated at 15K chars ...]"

    # Build changed file listing
    file_listing = []
    for cf in changed_files:
        deps = dependencies.get(cf, [])
        dep_str = f"  callers: {', '.join(deps[:5])}" if deps else ""
        file_listing.append(f"  {cf}{dep_str}")

    # Include headers of changed files for context
    headers = []
    for cf in changed_files[:10]:
        header = _read_file_header(project_path / cf, 30)
        if header:
            headers.append(f"--- {cf} ---\n{header}")

    header_block = "\n\n".join(headers) if headers else ""

    seed = f"""DIFF-AUDIT: Audit uncommitted changes in {project_path.name}

You are auditing the UNCOMMITTED CHANGES in this project. Focus on the diff
and its ripple effects -- not the entire codebase.

CHANGED FILES:
{chr(10).join(file_listing)}

DIFF:
{diff_content}
{f"\nFILE HEADERS (for context):\n{header_block}" if header_block else ""}
MODE-FORCING RULES:
1. Every issue MUST have: [SEVERITY] file:line -- description
2. Severity is one of: CRITICAL, HIGH, MEDIUM, LOW
3. Anchor your findings to exact diff lines (hunk headers show @@ ... @@)
4. Consider ripple effects: if the diff changes function foo(), check callers
5. Verify: are the changes correct? Do they break existing behavior?
6. Each issue: exact location, what's wrong, WHY it matters, concrete fix

YOUR APPROACH:
- Read the actual source files surrounding the changes
- Check that new code handles edge cases
- Verify error paths
- Trace symbol dependencies (callers/callees) for correctness

OUTPUT FORMAT (required):
For each issue:
[SEVERITY] path/to/file:line_number -- Description
  Impact: Why this matters
  Fix: Concrete fix or approach

End each iteration with:
TOTAL: X issues (Y CRITICAL, Z HIGH, W MEDIUM, V LOW)
NEW THIS ITERATION: N issues
{f"\nADDITIONAL CONTEXT:\n{extra_context}" if extra_context else ""}"""

    return seed


def _build_minimal_seed(project_path: Path, extra_context: str) -> str:
    """Fallback seed when no source files are found."""
    return f"""SELF-AUDIT: Audit project at {project_path}

No standard source files detected. Read the project files yourself and
find bugs, design flaws, and correctness issues.

MODE-FORCING RULES:
1. Every issue: [SEVERITY] file:line -- description
2. Read actual files. Cite real code.
3. Severity: CRITICAL, HIGH, MEDIUM, LOW
{f"\nADDITIONAL CONTEXT:\n{extra_context}" if extra_context else ""}"""
