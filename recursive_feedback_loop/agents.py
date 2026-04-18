"""Multi-agent configuration for roundtable RFL.

Defines AgentConfig — one AI participant in a roundtable loop.
Agents can be loaded from a YAML file or defined via CLI --agent flags.

Example YAML:
  agents:
    - name: auditor
      model: anthropic/claude-sonnet-4
      provider: anthropic
      role: Find security vulnerabilities and logic errors
    - name: optimizer
      model: google/gemini-2.5-pro
      provider: google
      role: Find performance bottlenecks and suggest optimizations
    - name: reviewer
      model: openai/glm-5.1
      provider: zai
      role: Find code style issues, missing tests, and documentation gaps
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List


@dataclass
class AgentConfig:
    """Configuration for a single AI agent in the roundtable."""

    name: str                          # short identifier (e.g. "auditor")
    model: Optional[str] = None        # hermes model (e.g. "anthropic/claude-sonnet-4")
    provider: Optional[str] = None     # hermes provider (e.g. "anthropic", "zai")
    role: str = ""                     # system-level role description
    profile: Optional[str] = None      # hermes profile

    def display(self) -> str:
        """Short display string for logging."""
        parts = [self.name]
        if self.model:
            parts.append(self.model)
        if self.provider:
            parts.append(f"via {self.provider}")
        return " ".join(parts)


def parse_agent_string(s: str) -> AgentConfig:
    """Parse an --agent string like 'name=foo,model=bar,role=baz'.

    Accepted keys: name, model, provider, role, profile
    name is required.
    """
    kv = {}
    # Split on commas, but allow commas inside quoted values
    # Simple approach: split on comma not inside quotes
    parts = re.split(r',\s*(?=[\w]+=)', s)
    for part in parts:
        part = part.strip()
        if '=' not in part:
            continue
        key, value = part.split('=', 1)
        key = key.strip().lower()
        value = value.strip().strip('"').strip("'")
        kv[key] = value

    name = kv.get('name')
    if not name:
        raise ValueError(f"--agent requires a 'name' field. Got: {s}")

    return AgentConfig(
        name=name,
        model=kv.get('model'),
        provider=kv.get('provider'),
        role=kv.get('role', ''),
        profile=kv.get('profile'),
    )


def load_agents_file(path: str) -> List[AgentConfig]:
    """Load agents from a YAML file.

    Expects a top-level 'agents' key with a list of agent dicts.
    Uses the same simple YAML parser as templates.py (key: value only).
    """
    text = Path(path).read_text().strip()
    agents = []

    # Find the agents: section
    in_agents = False
    current = {}

    for line in text.splitlines():
        stripped = line.strip()

        if not stripped or stripped.startswith('#'):
            continue

        # Top-level "agents:" line
        if stripped == 'agents:':
            in_agents = True
            continue

        if not in_agents:
            continue

        # New agent entry: starts with "- name:" or just "- "
        if stripped.startswith('- '):
            # Save previous agent if any
            if current.get('name'):
                agents.append(AgentConfig(
                    name=current['name'],
                    model=current.get('model'),
                    provider=current.get('provider'),
                    role=current.get('role', ''),
                    profile=current.get('profile'),
                ))
            current = {}
            # Parse the rest of the line after "- "
            after_dash = stripped[2:].strip()
            if '=' in after_dash:
                # Inline format: - name=foo,model=bar
                kv_parts = re.split(r',\s*(?=[\w]+=)', after_dash)
                for p in kv_parts:
                    if '=' in p:
                        k, v = p.split('=', 1)
                        current[k.strip().lower()] = v.strip().strip('"').strip("'")
            elif ':' in after_dash:
                # YAML format: - name: foo
                k, v = after_dash.split(':', 1)
                current[k.strip().lower()] = v.strip().strip('"').strip("'")
            continue

        # Indented key: value inside an agent block
        if ':' in stripped:
            k, v = stripped.split(':', 1)
            current[k.strip().lower()] = v.strip().strip('"').strip("'")

    # Don't forget the last agent
    if current.get('name'):
        agents.append(AgentConfig(
            name=current['name'],
            model=current.get('model'),
            provider=current.get('provider'),
            role=current.get('role', ''),
            profile=current.get('profile'),
        ))

    if not agents:
        raise ValueError(f"No agents found in {path}. Expected 'agents:' with a list of entries.")

    return agents


def validate_agents(agents: List[AgentConfig]) -> List[str]:
    """Validate agent configs. Returns list of warnings (empty = all good)."""
    warnings = []
    names = [a.name for a in agents]

    if len(names) != len(set(names)):
        dupes = [n for n in names if names.count(n) > 1]
        warnings.append(f"Duplicate agent names: {', '.join(set(dupes))}")

    for agent in agents:
        if not agent.model and not agent.provider:
            warnings.append(
                f"Agent '{agent.name}' has no model or provider specified. "
                f"Will use Hermes defaults."
            )
        if not agent.role:
            warnings.append(
                f"Agent '{agent.name}' has no role. "
                f"It will participate without a specific perspective."
            )

    return warnings
