"""OpenCode provider using CLI subprocess."""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import logging
import os
import re
import shutil
import tempfile
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

from agentfield.harness._cli import (
    estimate_cli_cost,
    extract_final_text,
    resolve_model_and_variant,
    extract_token_usage,
    parse_jsonl,
    run_cli,
    strip_ansi,
)
from agentfield.harness._availability import ensure_cli_available, provider_unavailable
from agentfield.harness._profiles import normalize_profile
from agentfield.harness._result import FailureType, Metrics, RawResult
from agentfield.exceptions import (
    HarnessProfileCapabilityError,
    HarnessProfileCleanupError,
    HarnessProfileError,
    HarnessProfileResolutionError,
)
from agentfield.types import ProfileId

logger = logging.getLogger("agentfield.harness.opencode")

# opencode CLI sometimes prints a hard error to stderr but exits 0
# (notably "Model not found", auth errors, schema validation failures).
# These patterns mark stderr as containing a real failure, not just noise
# like the one-time SQLite migration prelude.
_OPENCODE_STDERR_ERROR_PATTERNS = (
    re.compile(r"^Error:", re.MULTILINE),
    re.compile(r"\bModel not found\b"),
    re.compile(r"\bAuthenticationError\b"),
    re.compile(r"\bUnauthorized\b"),
    re.compile(r"\bAPIError\b"),
)


def _prompt_via_stdin() -> bool:
    """Whether to hand the prompt to opencode over stdin instead of argv.

    On Windows the CLI on PATH is usually an npm .cmd shim that runs via
    cmd.exe, whose ~8k command-line cap real prompts blow straight through
    ("The command line is too long."). opencode reads the prompt from stdin
    when the positional arg is absent, so feed it that way there. POSIX keeps
    the battle-tested positional-arg path.
    """
    return os.name == "nt"


def _count_turns_from_events(events: list[dict[str, object]]) -> int:
    """Count opencode turns from JSON events.

    Preferred definition is one turn per ``step_start`` event. If a stream
    has no step markers, fall back to counting ``tool_use`` events.
    """
    step_starts = sum(1 for event in events if event.get("type") == "step_start")
    if step_starts > 0:
        return step_starts

    tool_uses = sum(1 for event in events if event.get("type") == "tool_use")
    return tool_uses


def _cost_from_events(events: list[dict[str, object]]) -> float | None:
    """Sum opencode per-step costs when present in the JSON stream."""
    total_cost = 0.0
    found_cost = False

    for event in events:
        if event.get("type") != "step_finish":
            continue
        part = event.get("part")
        if not isinstance(part, dict):
            continue
        cost = part.get("cost")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            total_cost += float(cost)
            found_cost = True

    return total_cost if found_cost else None


def _tokens_from_events(
    events: list[dict[str, object]],
) -> tuple[dict[str, int], bool]:
    """Sum token counts from opencode ``step_finish.part.tokens`` objects."""
    total = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
    }
    found_tokens = False

    def _int(value: object) -> int:
        if isinstance(value, bool):
            return 0
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0

    for event in events:
        if event.get("type") != "step_finish":
            continue
        part = event.get("part")
        if not isinstance(part, dict):
            continue
        tokens = part.get("tokens")
        if not isinstance(tokens, dict):
            continue
        found_tokens = True
        total["input_tokens"] += _int(tokens.get("input"))
        total["output_tokens"] += _int(tokens.get("output")) + _int(
            tokens.get("reasoning")
        )
        cache = tokens.get("cache")
        if isinstance(cache, dict):
            total["cache_read_tokens"] += _int(cache.get("read"))
            total["cache_creation_tokens"] += _int(cache.get("write"))

    return total, found_tokens


def _extract_opencode_event_error(events: list[dict[str, object]]) -> str | None:
    """Pull a meaningful failure message from an in-band JSON error event."""
    for event in events:
        if event.get("type") != "error":
            continue

        for key in ("message", "error", "text"):
            value = event.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:1000]

        part = event.get("part")
        if isinstance(part, dict):
            for key in ("message", "error", "text"):
                value = part.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()[:1000]

        return str(event)[:1000]

    return None


def _extract_opencode_error(stderr: str) -> str:
    """Pull the meaningful failure line(s) out of opencode stderr.

    opencode's stderr typically opens with the SQLite migration prelude
    ("Performing one time database migration…") followed by the real error.
    Naively truncating the first 800 chars hides the part that matters,
    so prefer the line carrying the error marker plus a small window of
    context around it.
    """
    lines = stderr.splitlines()
    for i, line in enumerate(lines):
        for pat in _OPENCODE_STDERR_ERROR_PATTERNS:
            if pat.search(line):
                window = lines[max(0, i - 1) : i + 5]
                return "\n".join(window).strip()[:1000]
    return stderr[:1000]


@dataclass(frozen=True)
class OpenCodeCapabilities:
    """Executable capabilities required by the profile-managed adapter."""

    version: str
    supports_profiles: bool = True
    supports_agent_selection: bool = True
    supports_json_stream: bool = True
    supports_project_dir: bool = True
    supports_model_variant: bool = True


@dataclass(frozen=True)
class _ResolvedProfile:
    profile_id: ProfileId
    agent: dict[str, Any]
    base_config: dict[str, Any]
    source: str


CapabilityProbe = Callable[
    [str, Mapping[str, str], frozenset[str], Optional[str]],
    Awaitable[OpenCodeCapabilities] | OpenCodeCapabilities,
]

_PROFILE_CONFIG_FILENAME = "opencode.json"
_PROFILE_MANAGED_ENV_VAR = "AGENTFIELD_OPENCODE_PROFILE_MANAGED"
_PROFILE_REGISTRY_ENV_VAR = "AGENTFIELD_OPENCODE_PROFILES"
_PROFILE_FILE_ENV_VAR = "AGENTFIELD_OPENCODE_PROFILE_FILE"
_CAPABILITY_PROBE_TIMEOUT_SECONDS = 5.0
_SUPPORTED_OPEN_CODE_MAJOR = 1
_MIN_SUPPORTED_OPEN_CODE_MINOR = 18
_VERSION_RE = re.compile(r"\bv?(\d+)\.(\d+)(?:\.(\d+))?\b")

# These values are configuration selectors, not user credentials.  They are
# removed from the inherited environment before the generated values are
# applied so a caller cannot silently replace the profile policy.
_PROFILE_POLICY_ENV_VARS = frozenset(
    {
        "OPENCODE_CONFIG",
        "OPENCODE_CONFIG_DIR",
        "OPENCODE_CONFIG_CONTENT",
        "OPENCODE_PERMISSION",
        "OPENCODE_DISABLE_PROJECT_CONFIG",
        _PROFILE_MANAGED_ENV_VAR,
        _PROFILE_REGISTRY_ENV_VAR,
        _PROFILE_FILE_ENV_VAR,
    }
)

# AgentField control-plane connection and credential variables must not be
# handed to a child OpenCode session.  Provider credentials such as
# OPENROUTER_API_KEY and EXA_API_KEY intentionally remain inherited.
_AGENTFIELD_RUNTIME_ENV_VARS = frozenset(
    {
        "AGENTFIELD_API_KEY",
        "AGENTFIELD_API_AUTH_API_KEY",
        "AGENTFIELD_TOKEN",
        "AGENTFIELD_URL",
        "AGENTFIELD_BASE_URL",
        "AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN",
        "AGENTFIELD_AUTHORIZATION_ADMIN_TOKEN",
        "AGENTFIELD_AUTHORIZATION_MASTER_SEED",
        "AGENTFIELD_CONNECTOR_TOKEN",
        "AGENTFIELD_DATABASE_URL",
        "AGENTFIELD_POSTGRES_URL",
        "AGENTFIELD_STORAGE_POSTGRES_URL",
        "AGENTFIELD_STORAGE_POSTGRES_PASSWORD",
        "AGENTFIELD_JWT_SECRET",
        "AGENTFIELD_CONFIG_FILE",
        "AGENTFIELD_SERVER",
        "AGENTFIELD_SERVER_URL",
        "AGENTFIELD_CONTROL_PLANE_URL",
        "AGENTFIELD_CONTROL_PLANE_TOKEN",
        "AGENTFIELD_RUNTIME_URL",
        "AGENTFIELD_RUNTIME_TOKEN",
        "AGENTFIELD_RUNTIME_API_KEY",
        "AGENTFIELD_DID_PRIVATE_KEY",
        "AGENTFIELD_X25519_PRIVATE_KEY",
        "AGENTFIELD_PRIVATE_KEY",
        "AGENT_CALLBACK_URL",
        "AGENTFIELD_CALLBACK_URL",
    }
)
_PROFILE_UNSET_ENV_VARS = frozenset(
    _PROFILE_POLICY_ENV_VARS | _AGENTFIELD_RUNTIME_ENV_VARS
)

_AUTONOMOUS_PERMISSION_ACTIONS = (
    "read",
    "edit",
    "bash",
    "glob",
    "grep",
    "list",
    "todowrite",
    "webfetch",
    "websearch",
    "lsp",
    "skill",
)
_HEADLESS_DENIED_ACTIONS = ("task", "question", "external_directory", "doom_loop")
_READ_ONLY_PERMISSION_ACTIONS = {"read", "glob", "grep", "list", "lsp"}
_TOOL_ACTIONS = {
    "read": "read",
    "write": "edit",
    "edit": "edit",
    "patch": "edit",
    "bash": "bash",
    "glob": "glob",
    "grep": "grep",
    "list": "list",
    "todowrite": "todowrite",
    "webfetch": "webfetch",
    "websearch": "websearch",
    "lsp": "lsp",
    "skill": "skill",
    "task": "task",
    "question": "question",
}
_PROFILE_METADATA_KEYS = {
    "id",
    "name",
    "profile_id",
    "fallback",
    "is_fallback",
    "resolution",
    "selection",
    "selected",
    "selected_profile",
    "resolved_profile",
    "used_fallback",
    "source",
    "config",
    "agent",
    "permissions",
    "disable",
    "disabled",
    "type",
    "kind",
}


def _profile_error(
    error_type: type[HarnessProfileError],
    *,
    code: str,
    message: str,
    action: str,
    profile: ProfileId | None = None,
) -> HarnessProfileError:
    return error_type(
        "opencode",
        code=code,
        message=message,
        action=action,
        profile=str(profile) if profile is not None else None,
    )


def _string_env_overrides(options: Mapping[str, object]) -> dict[str, str]:
    value = options.get("env")
    if not isinstance(value, Mapping):
        return {}
    return {
        key: item
        for key, item in value.items()
        if isinstance(key, str) and isinstance(item, str)
    }


def _effective_options_env(options: Mapping[str, object]) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(_string_env_overrides(options))
    return environment


def _profile_mode_enabled(environment: Mapping[str, str]) -> bool:
    value = environment.get(_PROFILE_MANAGED_ENV_VAR, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _profile_management_requested(options: Mapping[str, object]) -> bool:
    """Whether this OpenCode request opts into profile-managed execution."""

    if normalize_profile(options.get("profile")) is not None:
        return True
    environment = _effective_options_env(options)
    return _profile_mode_enabled(environment) or any(
        bool(environment.get(name, "").strip())
        for name in (_PROFILE_REGISTRY_ENV_VAR, _PROFILE_FILE_ENV_VAR)
    )


def _read_profile_document(
    value: object,
    *,
    source: str,
    profile: ProfileId,
) -> tuple[dict[str, Any], str]:
    if isinstance(value, Mapping):
        try:
            return copy.deepcopy(dict(value)), source
        except Exception as exc:
            raise _profile_error(
                HarnessProfileResolutionError,
                code="profile_source_invalid",
                message="the configured profile source is not a serializable mapping",
                action="provide a JSON object containing the profile definitions",
                profile=profile,
            ) from exc
    if not isinstance(value, str) or not value.strip():
        raise _profile_error(
            HarnessProfileResolutionError,
            code="profile_source_invalid",
            message="the configured profile source must be a JSON object or file",
            action="provide a JSON object containing the profile definitions",
            profile=profile,
        )
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise _profile_error(
            HarnessProfileResolutionError,
            code="profile_source_invalid",
            message="the configured profile source is not valid JSON",
            action="fix the profile source and retry the run",
            profile=profile,
        ) from exc
    if not isinstance(parsed, dict):
        raise _profile_error(
            HarnessProfileResolutionError,
            code="profile_source_invalid",
            message="the configured profile source must contain a JSON object",
            action="provide a JSON object containing the profile definitions",
            profile=profile,
        )
    return parsed, source


def _read_profile_file(path: str, profile: ProfileId) -> tuple[dict[str, Any], str]:
    try:
        content = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise _profile_error(
            HarnessProfileResolutionError,
            code="profile_source_unreadable",
            message="the configured profile file could not be read",
            action="check the profile file path and permissions",
            profile=profile,
        ) from exc
    return _read_profile_document(content, source="profile file", profile=profile)


def _load_profile_document(
    options: Mapping[str, object],
    environment: Mapping[str, str],
    profile: ProfileId,
    *,
    profile_registry: Mapping[str, object] | None = None,
    profile_file: str | None = None,
) -> tuple[dict[str, Any], str]:
    if profile_registry is not None:
        return _read_profile_document(
            profile_registry, source="provider profile registry", profile=profile
        )

    for key in (
        "opencode_profile_registry",
        "profile_registry",
        "opencode_profiles",
        "profile_definitions",
    ):
        value = options.get(key)
        if value is not None:
            return _read_profile_document(
                value, source=f"option {key}", profile=profile
            )

    configured_file = profile_file
    if not configured_file:
        for key in ("opencode_profile_file", "profile_file"):
            value = options.get(key)
            if isinstance(value, str) and value.strip():
                configured_file = value
                break
    if not configured_file:
        configured_file = environment.get(_PROFILE_FILE_ENV_VAR)
    if isinstance(configured_file, str) and configured_file.strip():
        return _read_profile_file(configured_file, profile)

    inline_profiles = environment.get(_PROFILE_REGISTRY_ENV_VAR)
    if isinstance(inline_profiles, str) and inline_profiles.strip():
        return _read_profile_document(
            inline_profiles,
            source=_PROFILE_REGISTRY_ENV_VAR,
            profile=profile,
        )

    for key in ("opencode_config", "profile_config"):
        value = options.get(key)
        if value is not None:
            return _read_profile_document(
                value, source=f"option {key}", profile=profile
            )

    config_content = environment.get("OPENCODE_CONFIG_CONTENT")
    if isinstance(config_content, str) and config_content.strip():
        return _read_profile_document(
            config_content,
            source="OPENCODE_CONFIG_CONTENT",
            profile=profile,
        )

    config_path = environment.get("OPENCODE_CONFIG")
    if isinstance(config_path, str) and config_path.strip():
        return _read_profile_file(config_path, profile)

    config_dir = environment.get("OPENCODE_CONFIG_DIR")
    if isinstance(config_dir, str) and config_dir.strip():
        for filename in (_PROFILE_CONFIG_FILENAME, "opencode.jsonc", "config.json"):
            candidate = os.path.join(config_dir, filename)
            if os.path.isfile(candidate):
                return _read_profile_file(candidate, profile)

    raise _profile_error(
        HarnessProfileResolutionError,
        code="profile_definitions_missing",
        message="profile-managed mode has no profile definition source",
        action=(
            "configure AGENTFIELD_OPENCODE_PROFILE_FILE, "
            "AGENTFIELD_OPENCODE_PROFILES, or OPENCODE_CONFIG"
        ),
        profile=profile,
    )


def _looks_like_agent_config(value: Mapping[str, object]) -> bool:
    return any(
        key in value
        for key in (
            "model",
            "variant",
            "prompt",
            "system",
            "mode",
            "permission",
            "permissions",
            "tools",
            "description",
            "steps",
            "maxSteps",
        )
    )


def _profile_map(
    document: Mapping[str, Any], profile: ProfileId
) -> tuple[dict[str, object], dict[str, Any], dict[str, Any]]:
    """Extract profile entries, base config, and resolver metadata."""

    metadata: dict[str, Any] = {}
    for key in ("resolution", "selection"):
        if isinstance(document.get(key), Mapping):
            metadata.update(dict(document[key]))
    for key in (
        "selected",
        "selected_profile",
        "resolved_profile",
        "fallback",
        "is_fallback",
        "used_fallback",
        "source",
        "resolution",
    ):
        if key in document:
            metadata[key] = document[key]

    raw_profiles: object = document.get("profiles")
    if raw_profiles is None:
        raw_profiles = document.get("agent", document.get("agents"))

    base_config: dict[str, Any] = {}
    if isinstance(document.get("config"), Mapping):
        base_config = copy.deepcopy(dict(document["config"]))
    elif isinstance(raw_profiles, Mapping):
        base_config = copy.deepcopy(dict(document))
        base_config.pop("profiles", None)
    elif isinstance(document, Mapping) and not _looks_like_agent_config(document):
        # A mapping keyed directly by opaque profile ids is a compact registry.
        raw_profiles = document

    profiles: dict[str, object] = {}
    if isinstance(raw_profiles, Mapping):
        profiles = {
            str(key): copy.deepcopy(value) for key, value in raw_profiles.items()
        }
    elif isinstance(raw_profiles, list):
        for item in raw_profiles:
            if not isinstance(item, Mapping):
                continue
            identifier = item.get("profile_id", item.get("id", item.get("name")))
            if isinstance(identifier, str) and identifier:
                profiles[identifier] = copy.deepcopy(dict(item))

    if not profiles and _looks_like_agent_config(document):
        profiles[str(profile)] = copy.deepcopy(dict(document))
        base_config = {}

    return profiles, base_config, metadata


def _unwrap_profile_entry(
    raw_entry: object, profile: ProfileId
) -> dict[str, Any]:
    if not isinstance(raw_entry, Mapping):
        raise _profile_error(
            HarnessProfileResolutionError,
            code="profile_definition_missing",
            message="the selected profile has no object definition",
            action="add an object definition for the requested profile",
            profile=profile,
        )
    entry = copy.deepcopy(dict(raw_entry))
    nested = entry.get("config")
    if isinstance(nested, Mapping):
        wrapper = {
            key: value
            for key, value in entry.items()
            if key not in {"config", "agent"}
        }
        result = copy.deepcopy(dict(nested))
        for key, value in wrapper.items():
            result.setdefault(key, value)
        return result
    nested_agent = entry.get("agent")
    if isinstance(nested_agent, Mapping) and not _looks_like_agent_config(entry):
        result = copy.deepcopy(dict(nested_agent))
        for key, value in entry.items():
            if key != "agent":
                result.setdefault(key, value)
        return result
    return entry


def _permission_effect(value: object, profile: ProfileId) -> object:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "ask":
            return "deny"
        if normalized in {"allow", "deny"}:
            return normalized
        raise _profile_error(
            HarnessProfileResolutionError,
            code="permission_invalid",
            message="the profile contains an unsupported permission effect",
            action="use only allow, deny, or ask permission effects",
            profile=profile,
        )
    if isinstance(value, Mapping):
        return {
            str(key): _permission_effect(item, profile)
            for key, item in value.items()
        }
    raise _profile_error(
        HarnessProfileResolutionError,
        code="permission_invalid",
        message="the profile contains an invalid permission rule",
        action="use OpenCode permission rules with allow, deny, or ask effects",
        profile=profile,
    )


def _permission_rules(value: object, profile: ProfileId) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {
            str(key): _permission_effect(item, profile)
            for key, item in value.items()
        }
    if not isinstance(value, list):
        raise _profile_error(
            HarnessProfileResolutionError,
            code="permission_invalid",
            message="the profile permission definition must be an object or list",
            action="use OpenCode's permission object or rule list",
            profile=profile,
        )

    result: dict[str, object] = {}
    for rule in value:
        if not isinstance(rule, Mapping):
            raise _profile_error(
                HarnessProfileResolutionError,
                code="permission_invalid",
                message="the profile contains a non-object permission rule",
                action="remove invalid permission rules and retry",
                profile=profile,
            )
        action = rule.get("action", rule.get("tool"))
        effect = rule.get("effect", rule.get("decision"))
        if not isinstance(action, str) or not isinstance(effect, str):
            raise _profile_error(
                HarnessProfileResolutionError,
                code="permission_invalid",
                message="each profile permission rule needs an action and effect",
                action="provide action/effect permission rules",
                profile=profile,
            )
        translated_effect = _permission_effect(effect, profile)
        resource = rule.get("resource", "*")
        key = action.strip()
        if resource == "*" or resource == "":
            result[key] = translated_effect
            continue
        existing = result.get(key)
        if isinstance(existing, str):
            existing_rules: dict[str, object] = {"*": existing}
        elif isinstance(existing, dict):
            existing_rules = {
                str(rule_key): rule_value
                for rule_key, rule_value in existing.items()
            }
        else:
            existing_rules = {}
        existing_rules[str(resource)] = translated_effect
        result[key] = existing_rules
    return result


def _translated_permissions(
    entry: Mapping[str, Any], options: Mapping[str, object], profile: ProfileId
) -> dict[str, object]:
    permission_mode = str(options.get("permission_mode") or "auto").strip().lower()
    if permission_mode == "plan":
        permissions: dict[str, object] = {
            action: ("allow" if action in _READ_ONLY_PERMISSION_ACTIONS else "deny")
            for action in _AUTONOMOUS_PERMISSION_ACTIONS
        }
    else:
        permissions = {action: "allow" for action in _AUTONOMOUS_PERMISSION_ACTIONS}

    configured_tools = options.get("tools")
    if isinstance(configured_tools, list):
        enabled = {
            _TOOL_ACTIONS.get(str(tool).strip().lower())
            for tool in configured_tools
        }
        for action in _AUTONOMOUS_PERMISSION_ACTIONS:
            if action not in enabled:
                permissions[action] = "deny"

    configured_permission = entry.get("permission", entry.get("permissions"))
    if configured_permission is not None:
        permissions.update(_permission_rules(configured_permission, profile))

    configured_tools_map = entry.get("tools")
    if isinstance(configured_tools_map, Mapping):
        for tool, enabled in configured_tools_map.items():
            action = _TOOL_ACTIONS.get(str(tool).strip().lower())
            if action is None:
                continue
            permissions[action] = "allow" if bool(enabled) else "deny"

    # Profile-managed runs never wait for an interactive answer.  task and
    # question are explicitly denied even if a source profile requested ask.
    for action in _HEADLESS_DENIED_ACTIONS:
        permissions[action] = "deny"
    return permissions


def _materialized_agent(
    raw_entry: object, options: Mapping[str, object], profile: ProfileId
) -> dict[str, Any]:
    entry = _unwrap_profile_entry(raw_entry, profile)
    source = str(entry.get("source", "")).strip().lower()
    resolution = str(entry.get("resolution", "")).strip().lower()
    if any(
        bool(entry.get(key))
        for key in ("fallback", "is_fallback", "used_fallback")
    ) or source == "fallback" or resolution in {
        "fallback",
        "default",
    }:
        raise _profile_error(
            HarnessProfileResolutionError,
            code="profile_fallback_selected",
            message="profile resolution selected a fallback definition",
            action=(
                "select an exact primary profile; fallback resolution is not "
                "allowed"
            ),
            profile=profile,
        )

    mode = entry.get("mode", entry.get("kind", entry.get("type")))
    if isinstance(mode, str) and mode.strip().lower() in {"subagent", "sub-agent"}:
        raise _profile_error(
            HarnessProfileResolutionError,
            code="profile_subagent",
            message=(
                "the selected profile is a subagent and cannot be the primary "
                "run agent"
            ),
            action="select a profile with mode primary or all",
            profile=profile,
        )
    if mode is not None and str(mode).strip().lower() not in {"primary", "all"}:
        raise _profile_error(
            HarnessProfileResolutionError,
            code="profile_mode_invalid",
            message="the selected profile has an unsupported agent mode",
            action="select a profile with mode primary or all",
            profile=profile,
        )
    if bool(entry.get("disable", entry.get("disabled", False))):
        raise _profile_error(
            HarnessProfileResolutionError,
            code="profile_disabled",
            message="the selected profile is disabled",
            action="select an enabled primary profile",
            profile=profile,
        )

    agent = {
        key: copy.deepcopy(value)
        for key, value in entry.items()
        if key not in _PROFILE_METADATA_KEYS
    }
    agent.pop("permission", None)
    agent.pop("permissions", None)
    agent.pop("tools", None)
    agent["mode"] = "primary"
    agent["permission"] = _translated_permissions(entry, options, profile)
    return agent


def _resolve_profile_document(
    document: Mapping[str, Any],
    source: str,
    profile: ProfileId,
    options: Mapping[str, object],
) -> _ResolvedProfile:
    profiles, base_config, metadata = _profile_map(document, profile)
    fallback = metadata.get("fallback") or metadata.get("is_fallback") or metadata.get(
        "used_fallback"
    )
    selected = metadata.get(
        "selected_profile",
        metadata.get("selected", metadata.get("resolved_profile")),
    )
    if bool(fallback) or str(metadata.get("source", "")).strip().lower() == "fallback":
        raise _profile_error(
            HarnessProfileResolutionError,
            code="profile_fallback_selected",
            message="profile resolution selected a fallback definition",
            action=(
                "select an exact primary profile; fallback resolution is not "
                "allowed"
            ),
            profile=profile,
        )
    if isinstance(selected, str) and selected and selected != str(profile):
        raise _profile_error(
            HarnessProfileResolutionError,
            code="profile_not_selected",
            message="the requested profile was not the exact resolved selection",
            action=(
                "disable fallback selection and select the requested profile "
                "exactly"
            ),
            profile=profile,
        )

    key = str(profile)
    if key not in profiles:
        available = sorted(profiles)[:8]
        available_text = ", ".join(available) if available else "none"
        raise _profile_error(
            HarnessProfileResolutionError,
            code="profile_unknown",
            message=(
                "no exact profile definition was found "
                f"(available: {available_text})"
            ),
            action="register the requested profile before starting the run",
            profile=profile,
        )

    agent = _materialized_agent(profiles[key], options, profile)
    base_config = {
        str(config_key): copy.deepcopy(config_value)
        for config_key, config_value in base_config.items()
        if config_key
        not in {
            "agent",
            "agents",
            "profiles",
            "default_agent",
            "permission",
            "permissions",
            "tools",
            "resolution",
            "selection",
            "selected",
            "selected_profile",
            "resolved_profile",
            "fallback",
            "is_fallback",
            "used_fallback",
            "contract_version",
            "provider",
            "minimum_version",
            "run_surface",
        }
    }
    base_config.setdefault("$schema", "https://opencode.ai/config.json")
    return _ResolvedProfile(
        profile_id=profile,
        agent=agent,
        base_config=base_config,
        source=source,
    )


async def _probe_opencode_capabilities(
    bin_path: str,
    env: Mapping[str, str],
    unset_env: frozenset[str],
    cwd: Optional[str],
) -> OpenCodeCapabilities:
    """Probe an executable, rather than trusting a static version table."""

    try:
        version_stdout, version_stderr, version_code = await run_cli(
            [bin_path, "--version"],
            env=dict(env),
            cwd=cwd,
            timeout=_CAPABILITY_PROBE_TIMEOUT_SECONDS,
            unset_env=unset_env,
        )
        if version_code != 0:
            raise RuntimeError(version_stderr.strip() or "version command failed")
        version_text = (version_stdout or version_stderr).strip()
        match = _VERSION_RE.search(version_text)
        if match is None:
            raise RuntimeError("version output did not contain a supported version")
        major, minor = int(match.group(1)), int(match.group(2))
        version = match.group(0).lstrip("v")
        if (
            major != _SUPPORTED_OPEN_CODE_MAJOR
            or minor < _MIN_SUPPORTED_OPEN_CODE_MINOR
        ):
            raise HarnessProfileCapabilityError(
                "opencode",
                code="opencode_version_unsupported",
                message=(
                    f"executable version {version!r} is outside the supported "
                    "1.18+ 1.x profile surface"
                ),
                action="install OpenCode 1.18.x or a later supported 1.x release",
            )

        help_stdout, help_stderr, help_code = await run_cli(
            [bin_path, "run", "--help"],
            env=dict(env),
            cwd=cwd,
            timeout=_CAPABILITY_PROBE_TIMEOUT_SECONDS,
            unset_env=unset_env,
        )
        if help_code != 0:
            raise RuntimeError(help_stderr.strip() or "run help command failed")
        help_text = f"{help_stdout}\n{help_stderr}"
        required_flags = ("--agent", "--format", "--dir", "--model", "--variant")
        missing = [flag for flag in required_flags if flag not in help_text]
        if missing:
            raise HarnessProfileCapabilityError(
                "opencode",
                code="opencode_profile_surface_missing",
                message="the executable does not expose the required profile run flags",
                action=f"use an executable exposing: {', '.join(required_flags)}",
            )
        return OpenCodeCapabilities(version=version)
    except HarnessProfileCapabilityError:
        raise
    except TimeoutError as exc:
        raise HarnessProfileCapabilityError(
            "opencode",
            code="opencode_capability_probe_timeout",
            message="the executable capability probe timed out",
            action="verify the OpenCode executable is runnable and retry",
        ) from exc
    except (OSError, RuntimeError) as exc:
        raise HarnessProfileCapabilityError(
            "opencode",
            code="opencode_capability_probe_failed",
            message="the executable capability probe failed",
            action="install a supported OpenCode executable and retry",
        ) from exc


def _raw_result_from_run(
    *,
    stdout: str,
    stderr: str,
    returncode: int,
    model_value: Optional[str],
    effective_prompt: str,
    start_api: float,
) -> RawResult:
    """Map one completed OpenCode JSON stream to the shared result contract."""

    api_ms = int((time.monotonic() - start_api) * 1000)
    events = parse_jsonl(stdout)
    if events:
        result_text = extract_final_text(events)
    else:
        result_text = stdout.strip() if stdout.strip() else None
    event_error = _extract_opencode_event_error(events)
    clean_stderr = strip_ansi(stderr.strip()) if stderr else ""

    logger.info(
        "opencode finished: returncode=%d stdout=%d chars elapsed=%ds",
        returncode,
        len(stdout),
        api_ms // 1000,
    )
    if not result_text and clean_stderr:
        logger.warning("opencode no stdout. stderr: %s", clean_stderr[:800])

    if returncode < 0:
        failure_type = FailureType.CRASH
        is_error = True
        error_message: str | None = (
            f"Process killed by signal {-returncode}. stderr: {clean_stderr[:500]}"
            if clean_stderr
            else f"Process killed by signal {-returncode}."
        )
    elif returncode != 0 and result_text is None:
        failure_type = FailureType.CRASH
        is_error = True
        error_message = (
            _extract_opencode_error(clean_stderr)
            if clean_stderr
            else (f"Process exited with code {returncode} and produced no output.")
        )
    elif event_error is not None and result_text is None:
        failure_type = FailureType.CRASH
        is_error = True
        error_message = event_error
    elif (
        result_text is None
        and clean_stderr
        and any(pat.search(clean_stderr) for pat in _OPENCODE_STDERR_ERROR_PATTERNS)
    ):
        # OpenCode sometimes exits 0 on hard failures such as an invalid model.
        failure_type = FailureType.CRASH
        is_error = True
        error_message = _extract_opencode_error(clean_stderr)
    else:
        failure_type = FailureType.NONE
        is_error = False
        error_message = None

    stream_cost = _cost_from_events(events)
    if stream_cost is not None:
        estimated_cost = stream_cost
    else:
        estimated_cost = estimate_cli_cost(
            model=model_value or "",
            prompt=effective_prompt,
            result_text=result_text,
        )

    num_turns = _count_turns_from_events(events)
    if num_turns == 0 and result_text:
        num_turns = 1

    tokens, found_tokens = _tokens_from_events(events)
    if not found_tokens:
        tokens = extract_token_usage(events)

    return RawResult(
        result=result_text,
        messages=events,
        metrics=Metrics(
            duration_api_ms=api_ms,
            num_turns=num_turns,
            total_cost_usd=estimated_cost,
            session_id="",
            input_tokens=tokens["input_tokens"],
            output_tokens=tokens["output_tokens"],
            cache_read_tokens=tokens["cache_read_tokens"],
            cache_creation_tokens=tokens["cache_creation_tokens"],
            model=model_value,
        ),
        is_error=is_error,
        error_message=error_message,
        failure_type=failure_type,
        returncode=returncode,
    )


class OpenCodeProvider:
    """OpenCode CLI provider with an opt-in profile-managed execution path."""

    # Global concurrency limiter: prevents too many simultaneous opencode
    # processes from overwhelming the LLM API with concurrent requests.
    # Each opencode run spawns a full subprocess (pyright, DB migration, etc.)
    # so unbounded concurrency causes rate-limiting and transient failures.
    # Default raised 3 → 10 to match the typical pr-af review fan-out
    # (~6–8 review_dimension phases + 3 meta-lenses); OpenRouter handles
    # this comfortably on Kimi K2.6. Lower via OPENCODE_MAX_CONCURRENT if
    # your provider has tighter per-key rate limits.
    _MAX_CONCURRENT: ClassVar[int] = int(
        os.environ.get("OPENCODE_MAX_CONCURRENT", "10")
    )
    _concurrency_sem: ClassVar[Optional[asyncio.Semaphore]] = None

    # Shared XDG_DATA_HOME across calls when opt-in is enabled. SQLite
    # migrations only run once per process instead of per-call. None means
    # "fresh tempdir per call" (current default).
    _shared_data_dir: ClassVar[Optional[str]] = None
    _discarded_profile_dirs: ClassVar[set[str]] = set()

    def __init__(
        self,
        bin_path: str = "opencode",
        *,
        profile_registry: Mapping[str, object] | None = None,
        profile_file: str | None = None,
        capability_probe: CapabilityProbe | None = None,
    ):
        self._bin = bin_path
        self._profile_registry = profile_registry
        self._profile_file = profile_file
        self._capability_probe = capability_probe

    @classmethod
    def _get_semaphore(cls) -> asyncio.Semaphore:
        if cls._concurrency_sem is None:
            cls._concurrency_sem = asyncio.Semaphore(cls._MAX_CONCURRENT)
        return cls._concurrency_sem

    def _resolve_profile(self, options: Mapping[str, object]) -> _ResolvedProfile:
        profile = normalize_profile(options.get("profile"))
        if profile is None:
            raise _profile_error(
                HarnessProfileResolutionError,
                code="profile_missing",
                message="profile-managed OpenCode mode did not receive a profile",
                action="provide a non-empty opaque profile identifier",
            )
        if "\x00" in str(profile):
            raise _profile_error(
                HarnessProfileResolutionError,
                code="profile_id_invalid",
                message="the profile identifier contains a NUL character",
                action="provide a valid opaque profile identifier",
                profile=profile,
            )
        environment = _effective_options_env(options)
        document, source = _load_profile_document(
            options,
            environment,
            profile,
            profile_registry=self._profile_registry,
            profile_file=self._profile_file,
        )
        return _resolve_profile_document(document, source, profile, options)

    def validate_profile(self, options: Mapping[str, object]) -> None:
        """Resolve profiles synchronously before any child process is started."""

        if not _profile_management_requested(options):
            return
        profile = normalize_profile(options.get("profile"))
        if profile is None:
            raise _profile_error(
                HarnessProfileResolutionError,
                code="profile_missing",
                message="profile-managed OpenCode mode did not receive a profile",
                action="provide a non-empty opaque profile identifier",
            )
        resolved = self._resolve_profile(options)
        if isinstance(options, dict):
            options["_opencode_resolved_profile"] = resolved

    def profile_mode_requested(self, options: Mapping[str, object]) -> bool:
        """Expose provider-owned mode detection to the generic runner."""

        return _profile_management_requested(options)

    @classmethod
    def _remember_discarded_profile_dir(cls, path: str) -> None:
        cls._discarded_profile_dirs.add(os.path.abspath(path))

    @classmethod
    def _cleanup_profile_dir(cls, path: str) -> None:
        """Remove a generated config directory and permanently discard its path."""

        absolute = os.path.abspath(path)
        cls._remember_discarded_profile_dir(absolute)
        try:
            shutil.rmtree(absolute)
        except FileNotFoundError:
            return
        except Exception as exc:
            raise HarnessProfileCleanupError(
                "opencode",
                code="profile_cleanup_failed",
                message=(
                    "generated profile configuration cleanup failed; the "
                    "affected directory was discarded and will not be reused"
                ),
                action=(
                    "remove the affected temporary directory and check "
                    "temporary-directory permissions before retrying"
                ),
            ) from exc

    @staticmethod
    def _cleanup_data_dir(path: str) -> None:
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            return
        except Exception as exc:
            raise HarnessProfileCleanupError(
                "opencode",
                code="profile_cleanup_failed",
                message="OpenCode temporary data cleanup failed",
                action="check temporary-directory permissions before retrying",
            ) from exc

    def _materialize_profile(
        self, options: Mapping[str, object], resolved: _ResolvedProfile
    ) -> tuple[str, str, dict[str, str], frozenset[str]]:
        try:
            config_dir = tempfile.mkdtemp(prefix=".agentfield-opencode-profile-")
        except Exception as exc:
            raise _profile_error(
                HarnessProfileResolutionError,
                code="profile_materialization_failed",
                message="could not create isolated OpenCode profile configuration",
                action="check temporary-directory permissions and retry",
                profile=resolved.profile_id,
            ) from exc

        config_dir = os.path.abspath(config_dir)
        if config_dir in type(self)._discarded_profile_dirs:
            # This is only reachable with an unusual tempfile implementation or
            # a test fixture, but refusing it enforces the no-reuse invariant.
            try:
                type(self)._cleanup_profile_dir(config_dir)
            except HarnessProfileCleanupError:
                raise
            raise HarnessProfileCleanupError(
                "opencode",
                code="profile_directory_reused",
                message="an affected generated profile directory was selected again",
                action="use a fresh temporary directory before retrying",
            )
        config_path = os.path.join(config_dir, _PROFILE_CONFIG_FILENAME)
        payload = dict(resolved.base_config)
        payload.pop("agents", None)
        payload["agent"] = {str(resolved.profile_id): resolved.agent}
        payload["default_agent"] = str(resolved.profile_id)

        try:
            serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
            Path(config_path).write_text(serialized, encoding="utf-8")
            os.chmod(config_path, 0o600)
        except BaseException as exc:
            try:
                type(self)._cleanup_profile_dir(config_dir)
            except HarnessProfileCleanupError as cleanup_exc:
                raise cleanup_exc from exc
            if isinstance(exc, HarnessProfileError):
                raise
            raise _profile_error(
                HarnessProfileResolutionError,
                code="profile_materialization_failed",
                message="could not write isolated OpenCode profile configuration",
                action="check temporary-directory permissions and retry",
                profile=resolved.profile_id,
            ) from exc

        env = _string_env_overrides(options)
        for key in _AGENTFIELD_RUNTIME_ENV_VARS | _PROFILE_POLICY_ENV_VARS:
            env.pop(key, None)
        env["OPENCODE_CONFIG_DIR"] = config_dir
        env["OPENCODE_CONFIG"] = config_path
        env["OPENCODE_DISABLE_PROJECT_CONFIG"] = "1"
        return config_dir, config_path, env, _PROFILE_UNSET_ENV_VARS

    async def _ensure_profile_capabilities(
        self,
        env: Mapping[str, str],
        unset_env: frozenset[str],
        cwd: Optional[str],
    ) -> OpenCodeCapabilities:
        probe = self._capability_probe or _probe_opencode_capabilities
        probed = probe(self._bin, env, unset_env, cwd)
        result = await probed if inspect.isawaitable(probed) else probed
        if not isinstance(result, OpenCodeCapabilities):
            raise HarnessProfileCapabilityError(
                "opencode",
                code="opencode_capability_invalid",
                message="the executable capability probe returned an invalid result",
                action="provide a valid OpenCode capability fixture",
            )
        match = _VERSION_RE.search(result.version)
        if (
            match is None
            or int(match.group(1)) != _SUPPORTED_OPEN_CODE_MAJOR
            or int(match.group(2)) < _MIN_SUPPORTED_OPEN_CODE_MINOR
        ):
            raise HarnessProfileCapabilityError(
                "opencode",
                code="opencode_version_unsupported",
                message=(
                    "the executable capability fixture reported an unsupported "
                    "version"
                ),
                action="use OpenCode 1.18.x or a later supported 1.x release",
            )
        if not result.supports_profiles or not result.supports_agent_selection:
            raise HarnessProfileCapabilityError(
                "opencode",
                code="opencode_profiles_unsupported",
                message="the selected executable cannot honor profile agent selection",
                action="use an OpenCode executable with the profile run surface",
            )
        missing_capabilities = [
            name
            for name, supported in (
                ("JSON streaming", result.supports_json_stream),
                ("project directory selection", result.supports_project_dir),
                ("model variants", result.supports_model_variant),
            )
            if not supported
        ]
        if missing_capabilities:
            raise HarnessProfileCapabilityError(
                "opencode",
                code="opencode_profile_surface_missing",
                message=(
                    "the executable lacks required profile capabilities: "
                    + ", ".join(missing_capabilities)
                ),
                action=(
                    "use an OpenCode executable with the complete profile run "
                    "surface"
                ),
            )
        return result

    async def execute(self, prompt: str, options: dict[str, object]) -> RawResult:
        if _profile_management_requested(options):
            self.validate_profile(options)
        ensure_cli_available("opencode", self._bin)
        sem = self._get_semaphore()
        logger.debug(
            "Waiting for concurrency slot (%d/%d in use)",
            self._MAX_CONCURRENT - sem._value,
            self._MAX_CONCURRENT,
        )
        async with sem:
            return await self._execute_impl(prompt, options)

    async def _execute_profile_impl(
        self, prompt: str, options: dict[str, object]
    ) -> RawResult:
        """Run one profile-selected session with an isolated config directory."""

        resolved = options.get("_opencode_resolved_profile")
        if not isinstance(resolved, _ResolvedProfile):
            self.validate_profile(options)
            resolved = options.get("_opencode_resolved_profile")
        if not isinstance(resolved, _ResolvedProfile):  # defensive for custom mappings
            raise _profile_error(
                HarnessProfileResolutionError,
                code="profile_resolution_missing",
                message="the profile resolver did not return a profile definition",
                action="provide an exact primary profile definition and retry",
            )

        config_dir: Optional[str] = None
        temp_data_dir: Optional[str] = None
        cleanup_error: HarnessProfileCleanupError | None = None

        try:
            config_dir, _config_path, env, unset_env = self._materialize_profile(
                options, resolved
            )
            environment = _effective_options_env(options)
            reuse_data_dir = environment.get(
                "AGENTFIELD_OPENCODE_REUSE_DATA_DIR", "false"
            ).strip().lower() in ("1", "true", "yes")
            if reuse_data_dir:
                if type(self)._shared_data_dir is None or not os.path.isdir(
                    type(self)._shared_data_dir or ""
                ):
                    type(self)._shared_data_dir = tempfile.mkdtemp(
                        prefix=".secaf-opencode-data-shared-"
                    )
                data_dir = type(self)._shared_data_dir
            else:
                temp_data_dir = tempfile.mkdtemp(prefix=".secaf-opencode-data-")
                data_dir = temp_data_dir
            assert data_dir is not None
            env["XDG_DATA_HOME"] = data_dir

            root = options.get("project_dir") or options.get("cwd")

            profile_options = dict(options)
            profile_model = resolved.agent.get("model")
            if not isinstance(profile_model, str) or not profile_model.strip():
                profile_model = resolved.base_config.get("model")
            if not isinstance(options.get("model"), str) or not str(
                options.get("model")
            ).strip():
                if isinstance(profile_model, str) and profile_model.strip():
                    profile_options["model"] = profile_model
            if not isinstance(options.get("variant"), str) or not str(
                options.get("variant")
            ).strip():
                profile_variant = resolved.agent.get("variant")
                if isinstance(profile_variant, str) and profile_variant.strip():
                    profile_options["variant"] = profile_variant

            model_value, variant_value = resolve_model_and_variant(profile_options)
            cmd = [
                self._bin,
                "run",
                "--format",
                "json",
                "--agent",
                str(resolved.profile_id),
            ]
            if isinstance(root, str):
                cmd.extend(["--dir", root])
            if model_value:
                cmd.extend(["-m", model_value])
            if variant_value:
                cmd.extend(["--variant", variant_value])

            system_prompt = options.get("system_prompt")
            effective_prompt = prompt
            if isinstance(system_prompt, str) and system_prompt.strip():
                effective_prompt = (
                    f"SYSTEM INSTRUCTIONS:\n{system_prompt.strip()}\n\n"
                    f"---\n\nUSER REQUEST:\n{prompt}"
                )

            prompt_via_stdin = _prompt_via_stdin()
            if not prompt_via_stdin:
                cmd.append(effective_prompt)

            # Keep the legacy provider's process-cwd behavior: the project
            # root is selected with --dir, not by changing subprocess cwd.
            await self._ensure_profile_capabilities(env, unset_env, None)
            timeout_seconds = int(
                environment.get("AGENTFIELD_HARNESS_TIMEOUT_SECONDS", "1800")
            )
            start_api = time.monotonic()
            try:
                stdout, stderr, returncode = await run_cli(
                    cmd,
                    env=env,
                    cwd=None,
                    timeout=timeout_seconds,
                    input_text=effective_prompt if prompt_via_stdin else None,
                    unset_env=unset_env,
                )
            except FileNotFoundError as exc:
                raise provider_unavailable("opencode", self._bin) from exc
            except TimeoutError as exc:
                return RawResult(
                    is_error=True,
                    error_message=str(exc),
                    failure_type=FailureType.TIMEOUT,
                    metrics=Metrics(model=model_value),
                )

            return _raw_result_from_run(
                stdout=stdout,
                stderr=stderr,
                returncode=returncode,
                model_value=model_value,
                effective_prompt=effective_prompt,
                start_api=start_api,
            )
        finally:
            if temp_data_dir is not None:
                try:
                    type(self)._cleanup_data_dir(temp_data_dir)
                except HarnessProfileCleanupError as exc:
                    cleanup_error = exc
            if config_dir is not None:
                try:
                    type(self)._cleanup_profile_dir(config_dir)
                except HarnessProfileCleanupError as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
            if cleanup_error is not None:
                raise cleanup_error

    async def _execute_impl(self, prompt: str, options: dict[str, object]) -> RawResult:
        if _profile_management_requested(options):
            return await self._execute_profile_impl(prompt, options)

        # opencode v1.4+ uses the `run` subcommand (replaces deprecated -p/-c flags)
        cmd = [self._bin, "run"]
        cmd.extend(["--format", "json"])

        # --dir is the project root the agent may read and write. project_dir is
        # the canonical field; fall back to cwd when it is unset. Previously cwd
        # took precedence, so a nested cwd under a shared project_dir made
        # opencode treat the rest of the root as an external directory and
        # auto-reject reads/writes of sibling paths — see agentfield#684. This
        # now matches the Go SDK's opencode provider precedence.
        dir_value = options.get("project_dir") or options.get("cwd")
        if isinstance(dir_value, str):
            cmd.extend(["--dir", dir_value])

        # Pass model via -m. A "#variant" suffix on the model (or an explicit
        # options["variant"]) maps to --variant — opencode's provider-specific
        # reasoning effort (e.g. high, max, minimal).
        model_value, variant_value = resolve_model_and_variant(options)
        if model_value:
            cmd.extend(["-m", model_value])
        if variant_value:
            cmd.extend(["--variant", variant_value])

        # opencode v1.14 does not accept --dangerously-skip-permissions on the
        # `run` subcommand — passing it makes yargs print the run-help screen
        # to stdout and exit 0, which the SDK then captures as the LLM
        # response. opencode in non-TTY mode proceeds without permission
        # prompting, so no flag is needed. See agentfield#582.

        # Handle system prompt - prepend to user prompt since OpenCode
        # has no native --system-prompt flag
        effective_prompt = prompt
        system_prompt = options.get("system_prompt")
        if isinstance(system_prompt, str) and system_prompt.strip():
            effective_prompt = (
                f"SYSTEM INSTRUCTIONS:\n{system_prompt.strip()}\n\n"
                f"---\n\nUSER REQUEST:\n{prompt}"
            )

        # Prompt is a positional arg to `opencode run` (not -p) on POSIX; on
        # Windows it goes over stdin instead (see _prompt_via_stdin).
        prompt_via_stdin = _prompt_via_stdin()
        if not prompt_via_stdin:
            cmd.append(effective_prompt)

        env: Dict[str, str] = {}
        env_value = options.get("env")
        if isinstance(env_value, dict):
            env = {
                str(key): str(value)
                for key, value in env_value.items()
                if isinstance(key, str) and isinstance(value, str)
            }

        # Model is passed via -m flag on the run subcommand (see above)

        cwd: Optional[str] = None

        # Per-call XDG_DATA_HOME by default — guarantees session isolation.
        # AGENTFIELD_OPENCODE_REUSE_DATA_DIR=true reuses one dir across calls
        # in this process so SQLite migrations only run once. Opt-in because
        # the implications vary by container layout (read-only /tmp, multi-
        # tenant deployments, etc.) — default behavior is unchanged.
        reuse_data_dir = os.environ.get(
            "AGENTFIELD_OPENCODE_REUSE_DATA_DIR", "false"
        ).strip().lower() in ("1", "true", "yes")

        temp_data_dir: Optional[str] = None
        if reuse_data_dir:
            if type(self)._shared_data_dir is None or not os.path.isdir(
                type(self)._shared_data_dir or ""
            ):
                type(self)._shared_data_dir = tempfile.mkdtemp(
                    prefix=".secaf-opencode-data-shared-"
                )
            data_dir = type(self)._shared_data_dir
        else:
            temp_data_dir = tempfile.mkdtemp(prefix=".secaf-opencode-data-")
            data_dir = temp_data_dir
        assert data_dir is not None
        env["XDG_DATA_HOME"] = data_dir

        # Wall-clock cap for ONE opencode subprocess. Default 1800s (30min):
        # Kimi K2.6 on a complex review can need 20+ minutes; cutting at 600s
        # (the previous default) was killing slow-but-progressing calls and
        # then re-running them from scratch via the runner's transient retry
        # path. If a call doesn't finish in 30 min, the prompt or the model
        # is wrong — re-running won't help — so we let it fail cleanly.
        # Override via AGENTFIELD_HARNESS_TIMEOUT_SECONDS for tighter caps.
        timeout_seconds = int(
            os.environ.get("AGENTFIELD_HARNESS_TIMEOUT_SECONDS", "1800")
        )

        start_api = time.monotonic()

        try:
            try:
                stdout, stderr, returncode = await run_cli(
                    cmd,
                    env=env,
                    cwd=cwd,
                    timeout=timeout_seconds,
                    input_text=effective_prompt if prompt_via_stdin else None,
                )
            except FileNotFoundError as exc:
                raise provider_unavailable("opencode", self._bin) from exc
            except TimeoutError as exc:
                return RawResult(
                    is_error=True,
                    error_message=str(exc),
                    failure_type=FailureType.TIMEOUT,
                    metrics=Metrics(),
                )
        finally:
            # Only clean up the per-call tempdir; the shared one outlives the
            # call by design (skip the SQLite migration on the next call).
            if temp_data_dir is not None:
                shutil.rmtree(temp_data_dir, ignore_errors=True)

        api_ms = int((time.monotonic() - start_api) * 1000)
        events = parse_jsonl(stdout)
        if events:
            result_text = extract_final_text(events)
        else:
            result_text = stdout.strip() if stdout.strip() else None
        event_error = _extract_opencode_event_error(events)
        clean_stderr = strip_ansi(stderr.strip()) if stderr else ""

        logger.info(
            "opencode finished: returncode=%d stdout=%d chars elapsed=%ds",
            returncode,
            len(stdout),
            api_ms // 1000,
        )
        if not result_text and clean_stderr:
            logger.warning("opencode no stdout. stderr: %s", clean_stderr[:800])

        if returncode < 0:
            failure_type = FailureType.CRASH
            is_error = True
            error_message: str | None = (
                f"Process killed by signal {-returncode}. stderr: {clean_stderr[:500]}"
                if clean_stderr
                else f"Process killed by signal {-returncode}."
            )
        elif returncode != 0 and result_text is None:
            failure_type = FailureType.CRASH
            is_error = True
            error_message = (
                _extract_opencode_error(clean_stderr)
                if clean_stderr
                else (f"Process exited with code {returncode} and produced no output.")
            )
        elif event_error is not None and result_text is None:
            failure_type = FailureType.CRASH
            is_error = True
            error_message = event_error
        elif (
            result_text is None
            and clean_stderr
            and any(pat.search(clean_stderr) for pat in _OPENCODE_STDERR_ERROR_PATTERNS)
        ):
            # opencode sometimes exits 0 even on hard failures like
            # "Model not found" — surface the real error from stderr instead
            # of silently returning empty output that downstream callers
            # interpret as "agent failed to produce a valid result".
            failure_type = FailureType.CRASH
            is_error = True
            error_message = _extract_opencode_error(clean_stderr)
        else:
            failure_type = FailureType.NONE
            is_error = False
            error_message = None

        stream_cost = _cost_from_events(events)
        if stream_cost is not None:
            estimated_cost = stream_cost
        else:
            estimated_cost = estimate_cli_cost(
                model=model_value or "",
                prompt=effective_prompt,
                result_text=result_text,
            )

        num_turns = _count_turns_from_events(events)
        if num_turns == 0 and result_text:
            num_turns = 1

        tokens, found_tokens = _tokens_from_events(events)
        if not found_tokens:
            tokens = extract_token_usage(events)

        return RawResult(
            result=result_text,
            messages=events,
            metrics=Metrics(
                duration_api_ms=api_ms,
                num_turns=num_turns,
                total_cost_usd=estimated_cost,
                session_id="",
                input_tokens=tokens["input_tokens"],
                output_tokens=tokens["output_tokens"],
                cache_read_tokens=tokens["cache_read_tokens"],
                cache_creation_tokens=tokens["cache_creation_tokens"],
                model=model_value,
            ),
            is_error=is_error,
            error_message=error_message,
            failure_type=failure_type,
            returncode=returncode,
        )
