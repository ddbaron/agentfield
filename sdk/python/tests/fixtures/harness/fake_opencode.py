#!/usr/bin/env python3
"""Executable OpenCode capability fixture used by behavioral tests."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _record(argv: list[str], config: dict[str, object] | None = None) -> None:
    log_path = os.environ.get("FAKE_OPENCODE_LOG")
    if not log_path:
        return
    record = {
        "argv": argv,
        "env": {
            key: os.environ[key]
            for key in (
                "OPENCODE_CONFIG",
                "OPENCODE_CONFIG_DIR",
                "OPENCODE_CONFIG_CONTENT",
                "OPENCODE_PERMISSION",
                "OPENCODE_DISABLE_PROJECT_CONFIG",
                "FAKE_PROVIDER_KEY",
                "FAKE_INHERITED_VALUE",
                "AGENTFIELD_API_KEY",
                "AGENTFIELD_TOKEN",
                "AGENTFIELD_URL",
                "AGENTFIELD_SERVER",
                "AGENTFIELD_X25519_PRIVATE_KEY",
                "AGENTFIELD_RUNTIME_TOKEN",
                "AGENT_CALLBACK_URL",
            )
            if key in os.environ
        },
        "config": config,
    }
    with Path(log_path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _load_config() -> dict[str, object]:
    config_path = os.environ.get("OPENCODE_CONFIG")
    if not config_path:
        return {}
    with Path(config_path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise SystemExit("generated config is not an object")
    return value


def main() -> int:
    argv = sys.argv[1:]
    if argv == ["--version"]:
        _record(argv)
        print("opencode 1.18.0")
        return 0
    if argv == ["run", "--help"]:
        _record(argv)
        print("--agent --format --dir --model --variant")
        return 0
    if not argv or argv[0] != "run":
        print("unsupported fixture invocation", file=sys.stderr)
        return 2

    config = _load_config()
    agents = config.get("agent")
    default_agent = config.get("default_agent")
    if not isinstance(agents, dict) or not isinstance(default_agent, str):
        print("profile config is missing agent/default_agent", file=sys.stderr)
        return 3
    selected = agents.get(default_agent)
    if not isinstance(selected, dict) or selected.get("mode") != "primary":
        print("profile config did not select a primary agent", file=sys.stderr)
        return 4
    permission = selected.get("permission")
    if not isinstance(permission, dict):
        print("profile config is missing permission policy", file=sys.stderr)
        return 5
    if any(value == "ask" for value in permission.values()):
        print("interactive permission leaked into profile config", file=sys.stderr)
        return 6
    if permission.get("task") != "deny" or permission.get("question") != "deny":
        print("headless task/question policy is missing", file=sys.stderr)
        return 7
    _record(argv, config)
    print(json.dumps({"type": "step_start", "step": 1}))
    print(json.dumps({"type": "text", "part": {"text": "fixture result"}}))
    print(
        json.dumps(
            {
                "type": "step_finish",
                "part": {
                    "cost": 0.004,
                    "tokens": {
                        "input": 11,
                        "output": 7,
                        "reasoning": 2,
                        "cache": {"read": 3, "write": 1},
                    },
                },
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
