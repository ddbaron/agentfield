from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from agentfield import HarnessConfig, ProfileId
from agentfield.exceptions import (
    HarnessProfileCapabilityError,
    HarnessProfileCleanupError,
    HarnessProfileResolutionError,
    HarnessProfileUnsupportedError,
)
from agentfield.harness._runner import HarnessRunner
from agentfield.harness._result import RawResult
from agentfield.harness.providers.aforge import AforgeProvider
from agentfield.harness.providers import ProfileCapableProvider
from agentfield.harness.providers.claude import ClaudeCodeProvider
from agentfield.harness.providers.codex import CodexProvider
from agentfield.harness.providers.gemini import GeminiProvider
from agentfield.harness.providers.grok import GrokProvider
from agentfield.harness.providers.opencode import (
    OpenCodeCapabilities,
    OpenCodeProvider,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "harness"
FAKE_OPENCODE = FIXTURE_DIR / "fake_opencode.py"
CAPABILITY_FIXTURE = FIXTURE_DIR / "opencode-profile-capabilities.json"


def _fixture_registry() -> dict[str, Any]:
    return json.loads(CAPABILITY_FIXTURE.read_text(encoding="utf-8"))


def _run_records(log_path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _contains_value(value: object, expected: object) -> bool:
    if value == expected:
        return True
    if isinstance(value, dict):
        return any(_contains_value(item, expected) for item in value.values())
    if isinstance(value, list):
        return any(_contains_value(item, expected) for item in value)
    return False


@pytest.fixture(autouse=True)
def _mock_fixture_binary_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agentfield.harness._availability.shutil.which", lambda path: path
    )
    OpenCodeProvider._concurrency_sem = None
    OpenCodeProvider._shared_data_dir = None


def test_profile_id_is_public_and_harness_config_is_typed() -> None:
    config = HarnessConfig(provider="opencode", profile=ProfileId("fixture-primary"))

    assert config.profile == "fixture-primary"
    assert config.model_dump()["profile"] == "fixture-primary"
    assert HarnessConfig(provider="opencode").profile is None


def test_profile_is_included_in_runner_option_resolution() -> None:
    config = HarnessConfig(provider="opencode", profile=ProfileId("fixture-primary"))
    runner = HarnessRunner(config)

    options = runner._config.model_dump(exclude_none=True)

    assert options["profile"] == "fixture-primary"


def test_profile_capability_protocol_requires_only_validation() -> None:
    class ValidatingProvider:
        def validate_profile(self, options: Mapping[str, object]) -> None:
            _ = options

    assert isinstance(ValidatingProvider(), ProfileCapableProvider)


@pytest.mark.asyncio
async def test_harness_config_profile_reaches_opencode_provider(
    tmp_path: Path,
):
    log_path = tmp_path / "runner.jsonl"
    config = HarnessConfig(
        provider="opencode",
        profile=ProfileId("fixture-primary"),
        opencode_bin=str(FAKE_OPENCODE),
        opencode_profile_registry=_fixture_registry(),
        cwd=str(tmp_path),
        env={
            "FAKE_OPENCODE_LOG": str(log_path),
        },
    )

    result = await HarnessRunner(config).run("use the configured profile")

    assert result.is_error is False
    assert result.result == "fixture result"
    assert any(
        record["argv"][0] == "run" and len(record["argv"]) > 2
        for record in _run_records(log_path)
    )


@pytest.mark.asyncio
async def test_non_opencode_provider_rejects_profile_before_execute(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []

    class ProviderWithoutProfiles:
        async def execute(self, prompt: str, options: dict[str, object]) -> RawResult:
            calls.append(prompt)
            return RawResult(result="unexpected")

    monkeypatch.setattr(
        "agentfield.harness._runner.build_provider",
        lambda _config: ProviderWithoutProfiles(),
    )

    with pytest.raises(HarnessProfileUnsupportedError) as raised:
        await HarnessRunner().run(
            "do not launch",
            provider="codex",
            profile=ProfileId("fixture-primary"),
        )

    assert raised.value.code == "profile_unsupported"
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider",
    [
        AforgeProvider(),
        ClaudeCodeProvider(),
        CodexProvider(),
        GeminiProvider(),
        GrokProvider(),
    ],
)
async def test_direct_non_opencode_providers_reject_profile_before_launch(provider):
    with pytest.raises(HarnessProfileUnsupportedError):
        await provider.execute(
            "do not launch", {"profile": ProfileId("fixture-primary")}
        )


@pytest.mark.asyncio
async def test_profile_resolution_rejects_unknown_missing_subagent_and_fallback(
    tmp_path: Path,
):
    provider = OpenCodeProvider(str(FAKE_OPENCODE))
    registry = _fixture_registry()
    launch_log = tmp_path / "must-not-launch.jsonl"

    with pytest.raises(HarnessProfileResolutionError) as unknown:
        await provider.execute(
            "must not launch",
            {
                "profile": ProfileId("not-registered"),
                "profile_registry": registry,
                "env": {"FAKE_OPENCODE_LOG": str(launch_log)},
            },
        )
    assert unknown.value.code == "profile_unknown"

    with pytest.raises(HarnessProfileResolutionError) as missing:
        provider.validate_profile(
            {
                "env": {"AGENTFIELD_OPENCODE_PROFILE_MANAGED": "1"},
            }
        )
    assert missing.value.code == "profile_missing"

    with pytest.raises(HarnessProfileResolutionError) as subagent:
        provider.validate_profile(
            {"profile": ProfileId("fixture-subagent"), "profile_registry": registry}
        )
    assert subagent.value.code == "profile_subagent"

    with pytest.raises(HarnessProfileResolutionError) as fallback:
        provider.validate_profile(
            {"profile": ProfileId("fixture-fallback"), "profile_registry": registry}
        )
    assert fallback.value.code == "profile_fallback_selected"

    # Resolution is synchronous and precedes both binary probing and launch.
    assert not launch_log.exists()


@pytest.mark.asyncio
async def test_valid_profile_materializes_primary_default_and_headless_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    log_path = tmp_path / "opencode.jsonl"
    monkeypatch.setenv("AGENTFIELD_API_KEY", "runtime-secret")
    monkeypatch.setenv("AGENTFIELD_TOKEN", "runtime-token")
    monkeypatch.setenv("AGENTFIELD_URL", "http://control-plane")
    monkeypatch.setenv("AGENTFIELD_SERVER", "http://control-plane")
    monkeypatch.setenv("AGENTFIELD_RUNTIME_TOKEN", "runtime-token")
    monkeypatch.setenv("AGENTFIELD_X25519_PRIVATE_KEY", "private-key")
    monkeypatch.setenv("AGENT_CALLBACK_URL", "http://agent")
    monkeypatch.setenv("FAKE_INHERITED_VALUE", "inherited-value")

    raw = await OpenCodeProvider(str(FAKE_OPENCODE)).execute(
        "inspect the fixture",
        {
            "profile": ProfileId("fixture-primary"),
            "profile_registry": _fixture_registry(),
            "project_dir": str(tmp_path),
            "env": {
                "FAKE_OPENCODE_LOG": str(log_path),
                "FAKE_PROVIDER_KEY": "scoped-provider-credential",
            },
        },
    )

    assert raw.is_error is False
    assert raw.result == "fixture result"
    assert raw.metrics.total_cost_usd == pytest.approx(0.004)
    assert raw.metrics.input_tokens == 11
    assert raw.metrics.output_tokens == 9
    assert raw.metrics.cache_read_tokens == 3
    assert raw.metrics.cache_creation_tokens == 1

    records = _run_records(log_path)
    run_record = next(
        record
        for record in records
        if record["argv"][0] == "run" and len(record["argv"]) > 2
    )
    config = run_record["config"]
    assert config["default_agent"] == "fixture-primary"
    assert list(config["agent"]) == ["fixture-primary"]
    selected = config["agent"]["fixture-primary"]
    assert selected["mode"] == "primary"
    assert selected["permission"]["read"] == "allow"
    assert selected["permission"]["edit"] == "allow"
    assert selected["permission"]["bash"] == "deny"
    assert selected["permission"]["task"] == "deny"
    assert selected["permission"]["question"] == "deny"
    assert selected["permission"]["external_directory"] == "deny"
    assert selected["permission"]["doom_loop"] == "deny"
    assert not _contains_value(selected["permission"], "ask")

    env = run_record["env"]
    assert env["FAKE_PROVIDER_KEY"] == "scoped-provider-credential"
    assert env["FAKE_INHERITED_VALUE"] == "inherited-value"
    assert "AGENTFIELD_API_KEY" not in env
    assert "AGENTFIELD_TOKEN" not in env
    assert "AGENTFIELD_URL" not in env
    assert "AGENTFIELD_SERVER" not in env
    assert "AGENTFIELD_RUNTIME_TOKEN" not in env
    assert "AGENTFIELD_X25519_PRIVATE_KEY" not in env
    assert "AGENT_CALLBACK_URL" not in env
    assert env["OPENCODE_DISABLE_PROJECT_CONFIG"] == "1"
    assert "OPENCODE_CONFIG_CONTENT" not in env
    assert not Path(env["OPENCODE_CONFIG"]).exists()
    assert env["OPENCODE_CONFIG_DIR"] == str(Path(env["OPENCODE_CONFIG"]).parent)

    argv = run_record["argv"]
    assert argv[:3] == ["run", "--format", "json"]
    assert "--agent" in argv
    assert argv[argv.index("--agent") + 1] == "fixture-primary"
    assert argv[argv.index("--dir") + 1] == str(tmp_path)
    assert argv[argv.index("-m") + 1] == "openrouter/example/fixture"
    assert argv[argv.index("--variant") + 1] == "high"


@pytest.mark.asyncio
async def test_reserved_config_environment_values_cannot_override_generated_policy(
    tmp_path: Path,
):
    log_path = tmp_path / "reserved.jsonl"
    attacker_config = tmp_path / "attacker.json"
    attacker_config.write_text("{}", encoding="utf-8")
    provider = OpenCodeProvider(str(FAKE_OPENCODE))

    await provider.execute(
        "run with the selected profile",
        {
            "profile": ProfileId("fixture-primary"),
            "profile_registry": _fixture_registry(),
            "cwd": str(tmp_path / "nested"),
            "project_dir": str(tmp_path),
            "env": {
                "FAKE_OPENCODE_LOG": str(log_path),
                "OPENCODE_CONFIG": str(attacker_config),
                "OPENCODE_CONFIG_DIR": str(tmp_path / "attacker-dir"),
                "OPENCODE_CONFIG_CONTENT": json.dumps({"default_agent": "bad"}),
                "OPENCODE_PERMISSION": "ask",
            },
        },
    )

    run_record = next(
        record
        for record in _run_records(log_path)
        if record["argv"][0] == "run" and len(record["argv"]) > 2
    )
    env = run_record["env"]
    assert env["OPENCODE_CONFIG"] != str(attacker_config)
    assert env["OPENCODE_CONFIG_DIR"] != str(tmp_path / "attacker-dir")
    assert "OPENCODE_CONFIG_CONTENT" not in env
    assert "OPENCODE_PERMISSION" not in env
    assert run_record["argv"][run_record["argv"].index("--dir") + 1] == str(tmp_path)


@pytest.mark.asyncio
async def test_profile_runs_are_concurrent_and_configuration_isolation_is_per_run(
    tmp_path: Path,
):
    log_path = tmp_path / "concurrent.jsonl"
    provider = OpenCodeProvider(str(FAKE_OPENCODE))
    options = {
        "profile": ProfileId("fixture-primary"),
        "profile_registry": _fixture_registry(),
        "cwd": str(tmp_path),
        "env": {"FAKE_OPENCODE_LOG": str(log_path)},
    }

    first, second = await asyncio.gather(
        provider.execute("first", dict(options)),
        provider.execute("second", dict(options)),
    )

    assert first.result == "fixture result"
    assert second.result == "fixture result"
    run_records = [
        record
        for record in _run_records(log_path)
        if record["argv"][0] == "run" and len(record["argv"]) > 2
    ]
    config_dirs = {record["env"]["OPENCODE_CONFIG_DIR"] for record in run_records}
    assert len(run_records) == 2
    assert len(config_dirs) == 2
    assert all(not Path(path).exists() for path in config_dirs)


@pytest.mark.asyncio
async def test_capability_fixture_rejects_unsupported_version_before_run(
    tmp_path: Path,
):
    calls: list[list[str]] = []
    captured: dict[str, str] = {}

    async def probe(
        _bin: str,
        _env: Any,
        _unset: frozenset[str],
        _cwd: str | None,
    ) -> OpenCodeCapabilities:
        captured["config_dir"] = _env["OPENCODE_CONFIG_DIR"]
        return OpenCodeCapabilities(version="1.17.9")

    async def fake_run_cli(cmd, **_kwargs):
        calls.append(cmd)
        return "unexpected", "", 0

    provider = OpenCodeProvider(
        str(FAKE_OPENCODE), capability_probe=probe
    )
    original = __import__(
        "agentfield.harness.providers.opencode", fromlist=["run_cli"]
    ).run_cli
    module = __import__("agentfield.harness.providers.opencode", fromlist=["run_cli"])
    module.run_cli = fake_run_cli
    try:
        with pytest.raises(HarnessProfileCapabilityError) as raised:
            await provider.execute(
                "unsupported",
                {
                    "profile": ProfileId("fixture-primary"),
                    "profile_registry": _fixture_registry(),
                    "cwd": str(tmp_path),
                },
            )
    finally:
        module.run_cli = original

    assert raised.value.code == "opencode_version_unsupported"
    assert calls == []
    assert not Path(captured["config_dir"]).exists()


@pytest.mark.asyncio
async def test_profile_materialization_failure_cleans_up_setup_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    captured: list[str] = []
    module = __import__("agentfield.harness.providers.opencode", fromlist=["tempfile"])
    original_mkdtemp = module.tempfile.mkdtemp

    def capture_mkdtemp(*args, **kwargs):
        path = original_mkdtemp(*args, **kwargs)
        if kwargs.get("prefix", "").startswith(".agentfield-opencode-profile-"):
            captured.append(path)
        return path

    monkeypatch.setattr(module.tempfile, "mkdtemp", capture_mkdtemp)
    provider = OpenCodeProvider(str(FAKE_OPENCODE))
    registry = {
        "profiles": {
            "bad": {
                "mode": "primary",
                "not_json_serializable": object(),
            }
        }
    }

    with pytest.raises(HarnessProfileResolutionError) as raised:
        await provider.execute(
            "materialize",
            {
                "profile": ProfileId("bad"),
                "profile_registry": registry,
                "cwd": str(tmp_path),
            },
        )

    assert raised.value.code == "profile_materialization_failed"
    assert captured
    assert all(not Path(path).exists() for path in captured)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["provider", "timeout", "cancel"])
async def test_timeout_and_cancellation_cleanup_generated_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
):
    captured: dict[str, str] = {}

    async def probe(
        _bin: str,
        _env: Any,
        _unset: frozenset[str],
        _cwd: str | None,
    ) -> OpenCodeCapabilities:
        return OpenCodeCapabilities(version="1.18.0")

    async def fake_run_cli(cmd, **kwargs):
        _ = cmd
        captured["config_dir"] = kwargs["env"]["OPENCODE_CONFIG_DIR"]
        if failure == "provider":
            return "", "provider failure", 2
        if failure == "timeout":
            raise TimeoutError("fixture timed out")
        raise asyncio.CancelledError()

    monkeypatch.setattr("agentfield.harness.providers.opencode.run_cli", fake_run_cli)
    provider = OpenCodeProvider(str(FAKE_OPENCODE), capability_probe=probe)
    options = {
        "profile": ProfileId("fixture-primary"),
        "profile_registry": _fixture_registry(),
        "cwd": str(tmp_path),
    }

    if failure == "provider":
        result = await provider.execute("provider error", options)
        assert result.is_error is True
    elif failure == "timeout":
        result = await provider.execute("timeout", options)
        assert result.failure_type.value == "timeout"
    else:
        with pytest.raises(asyncio.CancelledError):
            await provider.execute("cancel", options)

    assert not Path(captured["config_dir"]).exists()


@pytest.mark.asyncio
async def test_cleanup_failure_is_redacted_and_next_run_does_not_reuse_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    captured_dirs: list[str] = []

    async def probe(
        _bin: str,
        _env: Any,
        _unset: frozenset[str],
        _cwd: str | None,
    ) -> OpenCodeCapabilities:
        return OpenCodeCapabilities(version="1.18.0")

    async def fake_run_cli(cmd, **kwargs):
        _ = cmd
        captured_dirs.append(kwargs["env"]["OPENCODE_CONFIG_DIR"])
        return "result", "", 0

    module = __import__("agentfield.harness.providers.opencode", fromlist=["shutil"])
    original_rmtree = module.shutil.rmtree

    def fail_generated_rmtree(path, *args, **kwargs):
        if ".agentfield-opencode-profile-" in str(path):
            raise OSError("sensitive temporary path")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr("agentfield.harness.providers.opencode.run_cli", fake_run_cli)
    monkeypatch.setattr(module.shutil, "rmtree", fail_generated_rmtree)
    provider = OpenCodeProvider(str(FAKE_OPENCODE), capability_probe=probe)
    options = {
        "profile": ProfileId("fixture-primary"),
        "profile_registry": _fixture_registry(),
        "cwd": str(tmp_path),
    }

    with pytest.raises(HarnessProfileCleanupError) as first:
        await provider.execute("first", dict(options))
    with pytest.raises(HarnessProfileCleanupError):
        await provider.execute("second", dict(options))

    assert first.value.code == "profile_cleanup_failed"
    assert "sensitive temporary path" not in str(first.value)
    assert len(captured_dirs) == 2
    assert captured_dirs[0] != captured_dirs[1]

    # The fixture intentionally made removal fail; restore the real remover for
    # test hygiene without weakening the provider assertion above.
    for path in captured_dirs:
        original_rmtree(path, ignore_errors=True)


class _Schema(BaseModel):
    answer: str


@pytest.mark.asyncio
async def test_runner_preserves_profile_for_schema_retry_provider_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class ProfileAwareProvider:
        def __init__(self) -> None:
            self.profile: object = None
            self.calls = 0

        def validate_profile(self, options: Mapping[str, object]) -> None:
            self.profile = options["profile"]

        async def execute(self, prompt: str, options: dict[str, object]) -> RawResult:
            _ = prompt
            self.calls += 1
            self.profile = options["profile"]
            output_path = next(
                part
                for part in prompt.split()
                if part.endswith(".agentfield_output.json")
            )
            payload = (
                {"wrong": "first attempt"}
                if self.calls == 1
                else {"answer": "ok"}
            )
            Path(output_path).write_text(json.dumps(payload), encoding="utf-8")
            return RawResult(result="ok")

    provider = ProfileAwareProvider()
    monkeypatch.setattr(
        "agentfield.harness._runner.build_provider", lambda _config: provider
    )

    result = await HarnessRunner().run(
        "write the result",
        provider="test-provider",
        profile=ProfileId("opaque-profile"),
        schema=_Schema,
        cwd=str(tmp_path),
        schema_max_retries=1,
    )

    assert result.is_error is False
    assert result.parsed == _Schema(answer="ok")
    assert provider.calls == 2
    assert provider.profile == "opaque-profile"
