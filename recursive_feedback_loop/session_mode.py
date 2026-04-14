"""Session mode — persistent Hermes session via tmux.

Instead of spawning a fresh Hermes per iteration (losing all context),
we keep one Hermes session alive in a tmux pane. Each iteration sends
a prompt and captures the response. Hermes retains its full conversation
history internally, so the compactor doesn't need to feed context back —
we just send a short nudge like "go deeper."

The compactor is still used for our own logging/export, but Hermes
doesn't need the compacted history because it lived through it.
"""

import subprocess
import time
import re
from typing import Optional


class HermesSession:
    """Manages a persistent Hermes CLI session in a tmux pane."""

    def __init__(
        self,
        session_name: str = "rfl_hermes",
        width: int = 200,
        height: int = 60,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        profile: Optional[str] = None,
        workdir: Optional[str] = None,
        startup_timeout: int = 15,
    ):
        self.session_name = session_name
        self.width = width
        self.height = height
        self.model = model
        self.provider = provider
        self.profile = profile
        self.workdir = workdir
        self.startup_timeout = startup_timeout
        self._ready = False
        self._last_capture = ""
        self._response_count = 0

    def start(self) -> bool:
        """Start the Hermes session in tmux. Returns True if ready."""
        cmd = f"hermes"
        if self.model:
            cmd += f" -m {self.model}"
        if self.provider:
            cmd += f" --provider {self.provider}"
        if self.profile:
            cmd += f" -p {self.profile}"

        full_cmd = f"tmux new-session -d -s {self.session_name} -x {self.width} -y {self.height} '{cmd}'"
        result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return False

        # Wait for Hermes to initialize and show the prompt
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            output = self._capture_pane()
            if "❯" in output:
                self._ready = True
                self._last_capture = output
                return True
            time.sleep(1)
        return False

    def is_ready(self) -> bool:
        """Check if the session is alive and at a prompt."""
        result = subprocess.run(
            f"tmux has-session -t {self.session_name}",
            shell=True, capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return False
        output = self._capture_pane()
        return "❯" in output.split("\n")[-5:]

    def send_prompt(self, prompt: str, timeout: int = 600) -> Optional[str]:
        """Send a prompt and wait for the response. Returns cleaned text."""
        if not self._ready and not self.is_ready():
            return None

        # Capture current state to know where the new response starts
        pre_capture = self._capture_pane()

        # Send the prompt
        # Use a unique marker so we can find where our response begins
        self._response_count += 1
        marker = f"[RFL-ITER-{self._response_count}]"

        # Send the prompt via tmux
        escaped = prompt.replace("'", "'\\''")
        subprocess.run(
            f"tmux send-keys -t {self.session_name} '{escaped}' Enter",
            shell=True, capture_output=True, text=True, timeout=10,
        )

        # Wait for response to complete (❯ prompt returns)
        deadline = time.time() + timeout
        last_change_time = time.time()

        while time.time() < deadline:
            time.sleep(3)
            current = self._capture_pane()

            # Check if the prompt indicator is back and Hermes is truly idle.
            # When Hermes is done: line is just "❯" alone.
            # When Hermes is still working: line is "⚕ ❯ type a message + Enter to interrupt..."
            last_lines = current.split("\n")[-5:]
            for line in last_lines:
                stripped = line.strip()
                # Done: "❯" alone on a line (not "⚕ ❯ type a message...")
                if stripped == "❯" and "type a message" not in current.split("\n")[-3:]:
                    # Make sure output actually changed (not just the initial state)
                    if current != pre_capture:
                        response = self._extract_response(pre_capture, current)
                        self._last_capture = current
                        return response

            # Track if output is still changing
            if current != self._last_capture:
                last_change_time = time.time()
                self._last_capture = current
            elif time.time() - last_change_time > 120:
                # No change for 2 minutes — probably stuck or finished silently
                response = self._extract_response(pre_capture, current)
                self._last_capture = current
                return response

        return None  # Timeout

    def _capture_pane(self) -> str:
        """Capture the full tmux pane content including scrollback."""
        try:
            result = subprocess.run(
                f"tmux capture-pane -t {self.session_name} -p -S -5000",
                shell=True, capture_output=True, text=True, timeout=10,
            )
            return result.stdout
        except Exception:
            return ""

    def _extract_response(self, pre_capture: str, post_capture: str) -> str:
        """Extract the new response text between two captures.

        Strategy: find the content that's new in post_capture vs pre_capture,
        then strip out the Hermes UI elements (boxes, tool calls, etc).
        """
        # Get lines that are new or changed
        pre_lines = pre_capture.split("\n")
        post_lines = post_capture.split("\n")

        # Find the first line that differs — that's where our response starts
        start_idx = 0
        min_len = min(len(pre_lines), len(post_lines))
        for i in range(min_len):
            if pre_lines[i] != post_lines[i]:
                start_idx = i
                break
        else:
            # All shared lines match — new content is appended
            start_idx = min_len

        new_lines = post_lines[start_idx:]

        # Clean the response
        return self._clean_response(new_lines)

    @staticmethod
    def _clean_response(lines: list) -> str:
        """Strip Hermes UI elements from response lines, keep content."""
        cleaned = []
        in_box = False
        box_content = []

        for line in lines:
            stripped = line.strip()

            # Skip the user prompt line (● ...)
            if stripped.startswith("● ") and not cleaned:
                continue

            # Skip "Initializing agent..."
            if "Initializing agent" in stripped:
                continue

            # Skip cogitating/ruminating lines
            if any(emoji in stripped for emoji in ["cogitating", "ruminating", "pondering", "thinking"]):
                continue

            # Skip "type a message" interrupt hint
            if "type a message" in stripped:
                continue

            # Skip interrupt messages
            if "New message detected" in stripped or "Interrupted during" in stripped:
                continue
            if "Sending after interrupt" in stripped:
                continue

            # Skip tool call progress (┊ lines)
            if "┊" in stripped and ("preparing" in stripped or "$" in stripped):
                continue

            # Detect Hermes response box
            if stripped.startswith("╭─") and "Hermes" in stripped:
                in_box = True
                continue
            if stripped.startswith("╰─") and in_box:
                in_box = False
                # Add accumulated box content
                if box_content:
                    cleaned.extend(box_content)
                    box_content = []
                continue

            if in_box:
                # Strip box-drawing side border (│ at start and end)
                content = stripped
                if content.startswith("│"):
                    content = content[1:]
                if content.endswith("│"):
                    content = content[:-1]
                content = content.strip()
                if content:
                    box_content.append(content)
                elif box_content:
                    # Empty line inside box — preserve paragraph break
                    box_content.append("")
                continue

            # Skip separator lines
            if stripped.startswith("───"):
                continue

            # Skip prompt line (❯)
            if stripped == "❯":
                continue

            # Skip status bar lines
            if "glm-" in stripped or "ctx" in stripped:
                continue

            # Skip empty trailing lines
            if not stripped and not cleaned:
                continue

            cleaned.append(line)

        result = "\n".join(cleaned).strip()

        # Remove leading/trailing empty lines
        while result.startswith("\n"):
            result = result[1:]
        while result.endswith("\n"):
            result = result[:-1]

        return result.strip()

    def stop(self):
        """Kill the tmux session."""
        subprocess.run(
            f"tmux kill-session -t {self.session_name} 2>/dev/null",
            shell=True, capture_output=True, text=True, timeout=5,
        )
        self._ready = False
