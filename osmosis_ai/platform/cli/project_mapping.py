from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from ipaddress import ip_address
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from filelock import BaseFileLock, FileLock

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.auth._fileutil import atomic_write_json
from osmosis_ai.platform.auth.config import PLATFORM_URL

SCHEMA_VERSION = 1
CONFIG_FILE = Path.home() / ".osmosis" / "config.json"
_HOST_LABEL_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")


@dataclass(frozen=True, slots=True)
class ProjectLinkRecord:
    project_path: str
    workspace_id: str
    workspace_name: str
    repo_url: str | None
    linked_at: str

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ProjectLinkRecord:
        try:
            project_path = _required_string(data, "projectPath")
            workspace_id = _required_string(data, "workspaceId")
            workspace_name = _required_string(data, "workspaceName")
            linked_at = _required_string(data, "linkedAt")
            return cls(
                project_path=project_path,
                workspace_id=workspace_id,
                workspace_name=workspace_name,
                repo_url=data.get("repoUrl")
                if isinstance(data.get("repoUrl"), str)
                else None,
                linked_at=linked_at,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CLIError("Invalid project link record") from exc

    def to_json(self) -> dict[str, Any]:
        data = {
            "projectPath": self.project_path,
            "workspaceId": self.workspace_id,
            "workspaceName": self.workspace_name,
            "linkedAt": self.linked_at,
        }
        if self.repo_url:
            repo_url = sanitize_repo_url(self.repo_url)
            if repo_url:
                data["repoUrl"] = repo_url
        return data


@dataclass(frozen=True, slots=True)
class ProjectMappingEntry:
    platform_key: str
    record: ProjectLinkRecord


class MappingConflictError(CLIError):
    pass


def now_linked_at() -> str:
    return datetime.now(UTC).isoformat()


def normalize_platform_key(platform_url: str | None = None) -> str:
    raw = platform_url or PLATFORM_URL
    try:
        parsed = urlsplit(raw)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise CLIError(f"Invalid Osmosis platform URL: {raw}") from exc
    if (
        not parsed.scheme
        or not parsed.netloc
        or hostname is None
        or not _is_valid_hostname(hostname)
    ):
        raise CLIError(f"Invalid Osmosis platform URL: {raw}")
    scheme = parsed.scheme.lower()
    port = None if port is not None and _is_default_port(scheme, port) else port
    netloc = _format_netloc(hostname, port)
    return urlunsplit((scheme, netloc, "", "", ""))


def sanitize_repo_url(repo_url: str | None) -> str | None:
    if not repo_url:
        return None
    raw = repo_url.strip()
    if "://" not in raw and "@" in raw and ":" in raw:
        user_host, path = raw.split(":", 1)
        user, _, host = user_host.partition("@")
        if not host or not path:
            return None
        if user != "git":
            return None
        if not _is_valid_hostname(host):
            return None
        return f"git@{host.lower()}:{path.split('?', 1)[0].split('#', 1)[0]}"
    try:
        parsed = urlsplit(raw)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme.lower() != "https"
        or hostname is None
        or not _is_valid_hostname(hostname)
    ):
        return None
    host = _format_netloc(hostname, port)
    return urlunsplit((parsed.scheme.lower(), host, parsed.path, "", ""))


def _is_valid_hostname(hostname: str | None) -> bool:
    if not hostname or any(char.isspace() for char in hostname):
        return False
    try:
        ip_address(hostname)
        return True
    except ValueError:
        pass
    labels = hostname.split(".")
    return all(label and _HOST_LABEL_RE.fullmatch(label) for label in labels)


def _format_netloc(hostname: str, port: int | None = None) -> str:
    host = hostname.lower()
    try:
        if ip_address(host).version == 6:
            host = f"[{host}]"
    except ValueError:
        pass
    if port is not None:
        return f"{host}:{port}"
    return host


def _is_default_port(scheme: str, port: int) -> bool:
    return (scheme == "http" and port == 80) or (scheme == "https" and port == 443)


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data[key]
    if not isinstance(value, str) or not value:
        raise ValueError(key)
    return value


class ProjectMappingStore:
    def __init__(
        self,
        *,
        config_file: Path = CONFIG_FILE,
        platform_url: str | None = None,
    ) -> None:
        self.config_file = config_file
        self.lock_file = config_file.with_suffix(f"{config_file.suffix}.lock")
        self.platform_key = normalize_platform_key(platform_url)

    def get_project(self, project_path: str) -> ProjectLinkRecord | None:
        with self._lock():
            data = self._read_unlocked()
            bucket = self._bucket(data)
            project_data = bucket["projects"].get(project_path)
            if not isinstance(project_data, dict):
                return None
            return ProjectLinkRecord.from_json(project_data)

    def list_projects(self) -> list[ProjectLinkRecord]:
        with self._lock():
            data = self._read_unlocked()
            bucket = self._bucket(data)
            return [
                ProjectLinkRecord.from_json(project_data)
                for _, project_data in sorted(bucket["projects"].items())
                if isinstance(project_data, dict)
            ]

    def list_all_projects(self) -> list[ProjectMappingEntry]:
        with self._lock():
            data = self._read_unlocked()
            platforms = data["platforms"]
            entries: list[ProjectMappingEntry] = []
            for platform_key, bucket in sorted(platforms.items()):
                if not isinstance(bucket, dict):
                    continue
                projects = bucket.get("projects")
                if not isinstance(projects, dict):
                    continue
                for project_path, project_data in sorted(projects.items()):
                    if not isinstance(project_data, dict):
                        raise CLIError(
                            f"Invalid project link record for {project_path}"
                        )
                    record = ProjectLinkRecord.from_json(project_data)
                    if record.project_path != project_path:
                        raise CLIError(f"Invalid project link path for {project_path}")
                    entries.append(
                        ProjectMappingEntry(platform_key=platform_key, record=record)
                    )
            return entries

    def check_link_allowed(self, record: ProjectLinkRecord) -> None:
        with self._lock():
            data = self._read_unlocked()
            bucket = self._bucket(data)
            self._check_link_allowed(bucket, record)

    def link(self, record: ProjectLinkRecord) -> ProjectLinkRecord:
        with self._lock():
            data = self._read_unlocked()
            bucket = self._bucket(data)
            self._check_link_allowed(bucket, record)
            stored = ProjectLinkRecord.from_json(record.to_json())
            bucket["projects"][stored.project_path] = stored.to_json()
            bucket["workspaceToProject"][stored.workspace_id] = stored.project_path
            self._write_unlocked(data)
            return stored

    def unlink(self, project_path: str) -> ProjectLinkRecord | None:
        with self._lock():
            data = self._read_unlocked()
            bucket = self._bucket(data)
            project_data = bucket["projects"].pop(project_path, None)
            if not isinstance(project_data, dict):
                return None
            record = ProjectLinkRecord.from_json(project_data)
            if bucket["workspaceToProject"].get(record.workspace_id) == project_path:
                del bucket["workspaceToProject"][record.workspace_id]
            self._write_unlocked(data)
            return record

    def update_workspace_cache(
        self,
        project_path: str,
        workspace_name: str,
        repo_url: str | None,
    ) -> ProjectLinkRecord | None:
        with self._lock():
            data = self._read_unlocked()
            bucket = self._bucket(data)
            project_data = bucket["projects"].get(project_path)
            if not isinstance(project_data, dict):
                return None
            current = ProjectLinkRecord.from_json(project_data)
            updated = ProjectLinkRecord(
                project_path=current.project_path,
                workspace_id=current.workspace_id,
                workspace_name=workspace_name,
                repo_url=sanitize_repo_url(repo_url),
                linked_at=current.linked_at,
            )
            bucket["projects"][project_path] = updated.to_json()
            bucket["workspaceToProject"][updated.workspace_id] = project_path
            self._write_unlocked(data)
            return updated

    def _default_data(self) -> dict[str, Any]:
        return {"version": SCHEMA_VERSION, "platforms": {}}

    def _lock(self) -> BaseFileLock:
        self.lock_file.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        return FileLock(str(self.lock_file))

    def _read_unlocked(self) -> dict[str, Any]:
        try:
            with self.config_file.open(encoding="utf-8") as file:
                data = json.load(file)
        except FileNotFoundError:
            return self._default_data()
        except json.JSONDecodeError as exc:
            raise CLIError(
                f"Invalid Osmosis config JSON at {self.config_file}"
            ) from exc
        except OSError as exc:
            raise CLIError(
                f"Unable to read Osmosis config at {self.config_file}: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise CLIError(f"Invalid Osmosis config at {self.config_file}")
        version = data.get("version", SCHEMA_VERSION)
        if not isinstance(version, int):
            raise CLIError(
                f"Invalid Osmosis config schema version at {self.config_file}"
            )
        if version > SCHEMA_VERSION:
            raise CLIError(
                "Osmosis config was written by a newer CLI. Please upgrade Osmosis."
            )
        data["version"] = version
        if not isinstance(data.get("platforms"), dict):
            data["platforms"] = {}
        return data

    def _bucket(self, data: dict[str, Any]) -> dict[str, Any]:
        platforms = data["platforms"]
        bucket = platforms.get(self.platform_key)
        if not isinstance(bucket, dict):
            bucket = {}
            platforms[self.platform_key] = bucket
        projects = bucket.get("projects")
        if not isinstance(projects, dict):
            projects = {}
            bucket["projects"] = projects
        bucket["workspaceToProject"] = self._rebuild_workspace_index(projects)
        return bucket

    def _rebuild_workspace_index(self, projects: dict[str, Any]) -> dict[str, str]:
        workspace_to_project: dict[str, str] = {}
        for project_path, project_data in projects.items():
            if not isinstance(project_data, dict):
                raise CLIError(f"Invalid project link record for {project_path}")
            record = ProjectLinkRecord.from_json(project_data)
            if record.project_path != project_path:
                raise CLIError(f"Invalid project link path for {project_path}")
            existing_path = workspace_to_project.get(record.workspace_id)
            if existing_path is not None and existing_path != project_path:
                raise MappingConflictError(
                    f"Workspace {record.workspace_name} ({record.workspace_id}) is already linked to {existing_path}"
                )
            workspace_to_project[record.workspace_id] = project_path
        return workspace_to_project

    def _write_unlocked(self, data: dict[str, Any]) -> None:
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.config_file, data, mode=0o600)

    def _check_link_allowed(
        self,
        bucket: dict[str, Any],
        record: ProjectLinkRecord,
    ) -> ProjectLinkRecord | None:
        project_data = bucket["projects"].get(record.project_path)
        if isinstance(project_data, dict):
            existing = ProjectLinkRecord.from_json(project_data)
            if existing.workspace_id == record.workspace_id:
                return existing
            raise MappingConflictError(
                f"Project {record.project_path} is already linked to "
                f"{existing.workspace_name} ({existing.workspace_id})"
            )

        existing_path = bucket["workspaceToProject"].get(record.workspace_id)
        if existing_path is not None and existing_path != record.project_path:
            raise MappingConflictError(
                f"Workspace {record.workspace_name} ({record.workspace_id}) is already linked to {existing_path}"
            )
        return None
