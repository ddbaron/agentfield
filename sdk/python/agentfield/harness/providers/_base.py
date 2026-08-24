from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agentfield.harness._result import RawResult


@runtime_checkable
class HarnessProvider(Protocol):
    async def execute(self, prompt: str, options: dict[str, object]) -> "RawResult": ...


@runtime_checkable
class ProfileCapableProvider(Protocol):
    """Optional provider capability for validating opaque profile requests."""

    def validate_profile(self, options: Mapping[str, object]) -> None: ...
