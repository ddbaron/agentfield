"""Provider-neutral harness profile contract helpers.

The AgentField surface deliberately knows only that a profile is an opaque
identifier.  Providers decide how that identifier is resolved and materialized
through their own capability surface.  Keeping these helpers here prevents the
runner from importing a provider SDK or learning provider-specific role names.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping

from agentfield.exceptions import (
    HarnessProfileResolutionError,
    HarnessProfileUnsupportedError,
)
from agentfield.types import ProfileId


def normalize_profile(value: object) -> ProfileId | None:
    """Classify a value for mode detection without raising.

    Invalid values return ``None`` here because this helper is also used when
    deciding whether an absent profile should keep the legacy path. Explicit
    public inputs must go through :func:`validate_profile_id`, which rejects
    invalid values instead of silently selecting profileless execution. No
    provider-specific normalization is performed.
    """

    if value is None:
        return None
    if not isinstance(value, str):
        return None
    if not value.strip():
        return None
    return ProfileId(value)


def validate_profile_id(
    value: object,
    *,
    provider: str,
    required: bool = False,
) -> ProfileId | None:
    """Validate a caller-supplied opaque profile identifier.

    ``normalize_profile`` is useful for detecting the legacy profileless path,
    but it intentionally does not raise.  Public call paths must use this
    helper as well so an explicitly supplied empty or non-string value cannot
    quietly turn into a profileless run.
    """

    if value is None:
        if required:
            raise HarnessProfileResolutionError(
                provider,
                code="profile_missing",
                message="profile-managed mode did not receive a profile",
                action="provide a non-empty opaque profile identifier",
            )
        return None

    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise HarnessProfileResolutionError(
            provider,
            code="profile_id_invalid",
            message="the profile identifier must be a non-empty string",
            action="provide a non-empty string profile identifier without NUL bytes",
        )

    return ProfileId(value)


def reject_profile_for_provider(provider: str, options: Mapping[str, object]) -> None:
    """Reject a non-empty profile unless the provider implements validation."""

    profile = validate_profile_id(options.get("profile"), provider=provider)
    if profile is not None:
        raise HarnessProfileUnsupportedError(
            provider,
            profile=str(profile),
            code="profile_unsupported",
            message="this provider does not advertise a profile capability",
            action=(
                "remove the profile or select a provider that implements the "
                "profile contract"
            ),
        )


async def validate_provider_profile(
    provider_name: str,
    provider: object,
    options: Mapping[str, object],
) -> None:
    """Validate profile capability before the provider launches a process.

    Providers that support profiles expose a synchronous or asynchronous
    ``validate_profile(options)`` method.  A provider may also expose
    ``profile_mode_requested(options)`` when it has provider-owned configuration
    that requires a profile even though the generic option is missing. All
    other providers reject a non-empty profile rather than silently dropping it.
    """

    validator = getattr(provider, "validate_profile", None)
    profile = validate_profile_id(options.get("profile"), provider=provider_name)
    mode_checker = getattr(provider, "profile_mode_requested", None)
    managed = profile is not None or (
        callable(mode_checker) and bool(mode_checker(options))
    )
    if not managed:
        return
    if callable(validator):
        result = validator(options)
        if inspect.isawaitable(result):
            await result
        return

    if profile is not None:
        raise HarnessProfileUnsupportedError(
            provider_name,
            profile=str(profile),
            code="profile_unsupported",
            message="this provider does not advertise a profile capability",
            action=(
                "remove the profile or select a provider that implements the "
                "profile contract"
            ),
        )

    raise HarnessProfileUnsupportedError(
        provider_name,
        code="profile_mode_unsupported",
        message=(
            "profile-managed mode was requested but this provider cannot "
            "resolve profiles"
        ),
        action="remove profile-managed configuration or select a capable provider",
    )


__all__ = [
    "normalize_profile",
    "validate_profile_id",
    "reject_profile_for_provider",
    "validate_provider_profile",
]
