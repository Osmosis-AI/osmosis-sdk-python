"""TOML config loading and validation for `osmosis rollout serve` runs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import tomllib
from pydantic import BaseModel, Field, field_validator

from osmosis_ai.cli.errors import CLIError

LogLevel = Literal["critical", "error", "warning", "info", "debug", "trace"]


class _ServeSection(BaseModel):
    rollout: str
    entrypoint: str


class _ServerSection(BaseModel):
    port: Annotated[int, Field(ge=1, le=65535)] = 9000
    host: str = "0.0.0.0"
    log_level: LogLevel = "info"


class _RegistrationSection(BaseModel):
    skip: bool = False
    api_key: str | None = None

    @field_validator("api_key", mode="before")
    @classmethod
    def _normalize_registration_api_key(cls, v: object) -> str | None:
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("registration.api_key must be a string or null")
        s = v.strip()
        return s if s else None


class _DebugSection(BaseModel):
    no_validate: bool = False
    trace_dir: str | None = None


class ServeConfig(BaseModel):
    """Parsed serve TOML configuration with flattened section fields."""

    serve_rollout: str
    serve_entrypoint: str
    server_port: Annotated[int, Field(ge=1, le=65535)]
    server_host: str
    server_log_level: LogLevel
    registration_skip: bool
    registration_api_key: str | None
    debug_no_validate: bool
    debug_trace_dir: str | None


def load_serve_config(path: Path) -> ServeConfig:
    """Load and validate TOML config for serve. Raises :class:`CLIError` on any problem."""
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except FileNotFoundError:
        raise CLIError(f"Config file not found: {path}") from None
    except tomllib.TOMLDecodeError as e:
        raise CLIError(f"Invalid TOML in {path}: {e}") from e
    except OSError as e:
        raise CLIError(f"Cannot read config file {path}: {e}") from e

    if "serve" not in raw:
        raise CLIError(f"Missing [serve] section in {path}")

    serve_section = raw["serve"]
    if not isinstance(serve_section, dict):
        raise CLIError(f"[serve] must be a table in {path}")

    for required_key in ("rollout", "entrypoint"):
        if required_key not in serve_section:
            raise CLIError(f"Missing '{required_key}' in [serve] section of {path}")

    server_section = raw.get("server", {})
    registration_section = raw.get("registration", {})
    debug_section = raw.get("debug", {})

    try:
        serve_parsed = _ServeSection(**serve_section)
        server = _ServerSection(**server_section)
        registration = _RegistrationSection(**registration_section)
        debug = _DebugSection(**debug_section)
    except Exception as e:
        raise CLIError(f"Invalid config in {path}: {e}") from e

    return ServeConfig(
        serve_rollout=serve_parsed.rollout,
        serve_entrypoint=serve_parsed.entrypoint,
        server_port=server.port,
        server_host=server.host,
        server_log_level=server.log_level,
        registration_skip=registration.skip,
        registration_api_key=registration.api_key,
        debug_no_validate=debug.no_validate,
        debug_trace_dir=debug.trace_dir,
    )


__all__ = ["ServeConfig", "load_serve_config"]
