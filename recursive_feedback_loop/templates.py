"""Template system for RFL — pre-built seed prompts and configs for common task types.

Templates live in two places:
  1. Built-in: recursive_feedback_loop/templates/<name>/template.yaml
  2. User:     ~/.config/rfl/templates/<name>/template.yaml

A template.yaml contains:
  name, description, config defaults (mode, iterations, budget, etc.),
  and a seed prompt with {{placeholder}} variables.

Usage:
  rfl run --template audit -w /path/to/project
  rfl run --template feature -w /path/to/project -p feature_name="infinite map"
  rfl templates                    # list available
  rfl templates audit              # show details + seed prompt
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Built-in templates directory
_BUILTIN_DIR = Path(__file__).parent / "templates"
# User templates directory
_USER_DIR = Path.home() / ".config" / "rfl" / "templates"


@dataclass
class Template:
    """A pre-built RFL template with seed prompt and config defaults."""
    name: str
    description: str
    seed_prompt: str
    # Config defaults (override what CLI provides)
    mode: Optional[str] = None
    iterations: Optional[int] = None
    budget: Optional[int] = None
    strategy: Optional[str] = None
    no_tools: Optional[bool] = None
    timeout: Optional[int] = None
    # Placeholders found in the seed prompt
    placeholders: list = field(default_factory=list)
    # Where this template was loaded from
    source: str = "builtin"


def _extract_placeholders(text: str) -> list:
    """Extract {{placeholder}} names from a template string."""
    return list(set(re.findall(r'\{\{(\w+)\}\}', text)))


def _fill_placeholders(text: str, params: dict) -> str:
    """Replace {{placeholder}} values in a template string.
    
    Leaves unreplaced placeholders as-is (they might be optional).
    """
    for key, value in params.items():
        text = text.replace(f"{{{{{key}}}}}", str(value))
    return text


def load_template(name: str) -> Optional[Template]:
    """Load a template by name. Checks user dir first, then built-in."""
    for base_dir, source in [(_USER_DIR, "user"), (_BUILTIN_DIR, "builtin")]:
        tmpl_dir = base_dir / name
        yaml_path = tmpl_dir / "template.yaml"
        seed_path = tmpl_dir / "seed.md"
        
        if not tmpl_dir.is_dir():
            continue
        
        # Read seed prompt from seed.md or inline from template.yaml
        seed = ""
        if seed_path.exists():
            seed = seed_path.read_text().strip()
        
        # Read template.yaml for config
        config = {}
        if yaml_path.exists():
            config = _parse_simple_yaml(yaml_path.read_text())
        
        # If seed wasn't in seed.md, check template.yaml
        if not seed and "seed_prompt" in config:
            seed = config.pop("seed_prompt", "")
        
        if not seed:
            continue
        
        tmpl = Template(
            name=config.get("name", name),
            description=config.get("description", ""),
            seed_prompt=seed,
            mode=config.get("mode"),
            iterations=int(config["iterations"]) if "iterations" in config else None,
            budget=int(config["budget"]) if "budget" in config else None,
            strategy=config.get("strategy"),
            no_tools=config.get("no_tools") == "true" if "no_tools" in config else None,
            timeout=int(config["timeout"]) if "timeout" in config else None,
            placeholders=_extract_placeholders(seed),
            source=source,
        )
        return tmpl
    
    return None


def list_templates() -> list:
    """List all available templates (user + built-in)."""
    seen = {}
    
    # User templates override built-ins with same name
    for base_dir, source in [(_BUILTIN_DIR, "builtin"), (_USER_DIR, "user")]:
        if not base_dir.is_dir():
            continue
        for tmpl_dir in sorted(base_dir.iterdir()):
            if not tmpl_dir.is_dir():
                continue
            yaml_path = tmpl_dir / "template.yaml"
            seed_path = tmpl_dir / "seed.md"
            if not yaml_path.exists() and not seed_path.exists():
                continue
            
            config = {}
            if yaml_path.exists():
                config = _parse_simple_yaml(yaml_path.read_text())
            
            seed = ""
            if seed_path.exists():
                seed = seed_path.read_text().strip()
            elif "seed_prompt" in config:
                seed = config["seed_prompt"]
            
            seen[tmpl_dir.name] = Template(
                name=config.get("name", tmpl_dir.name),
                description=config.get("description", "(no description)"),
                seed_prompt=seed,
                placeholders=_extract_placeholders(seed) if seed else [],
                source=source,
            )
    
    return list(seen.values())


def apply_template(tmpl: Template, params: dict = None) -> dict:
    """Apply a template, returning seed_prompt and config overrides.
    
    Returns: {"seed_prompt": str, "config_overrides": dict}
    """
    params = params or {}
    seed = _fill_placeholders(tmpl.seed_prompt, params)
    
    config_overrides = {}
    if tmpl.mode is not None:
        config_overrides["mode"] = tmpl.mode
    if tmpl.iterations is not None:
        config_overrides["iterations"] = tmpl.iterations
    if tmpl.budget is not None:
        config_overrides["budget"] = tmpl.budget
    if tmpl.strategy is not None:
        config_overrides["strategy"] = tmpl.strategy
    if tmpl.no_tools is not None:
        config_overrides["no_tools"] = tmpl.no_tools
    if tmpl.timeout is not None:
        config_overrides["timeout"] = tmpl.timeout
    
    return {
        "seed_prompt": seed,
        "config_overrides": config_overrides,
    }


def _parse_simple_yaml(text: str) -> dict:
    """Parse a dead-simple YAML subset (key: value pairs only, no nesting).
    
    Handles:
      key: value
      key: "quoted value"
      key: 123
      
    Does NOT handle arrays, nested objects, or multiline strings.
    That's intentional -- templates should be simple.
    """
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r'^(\w+):\s*(.*)', line)
        if match:
            key = match.group(1)
            value = match.group(2).strip()
            # Strip quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            result[key] = value
    return result
