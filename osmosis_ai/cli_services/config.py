from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import yaml
from yaml.representer import SafeRepresenter

from .errors import CLIError
from .shared import coerce_optional_float


@dataclass(frozen=True)
class ParsedItem:
    label: Optional[str]
    payload: Any


@dataclass(frozen=True)
class RubricConfig:
    rubric_id: str
    rubric_text: str
    model_info: dict[str, Any]
    score_min: Optional[float]
    score_max: Optional[float]
    system_prompt: Optional[str]
    original_input: Optional[str]
    ground_truth: Optional[str]
    source_label: str


@dataclass(frozen=True)
class RubricSuite:
    source_path: Path
    version: Optional[int]
    configs: dict[str, RubricConfig]

    def get(self, rubric_id: str) -> RubricConfig:
        if rubric_id not in self.configs:
            available = ", ".join(self.available_ids()) or "none"
            raise CLIError(
                f"Rubric '{rubric_id}' not found in '{self.source_path}'. Available IDs: {available}"
            )
        return self.configs[rubric_id]

    def available_ids(self) -> list[str]:
        return sorted(self.configs)


@dataclass(frozen=True)
class RubricConfigDocumentResult:
    configs: dict[str, RubricConfig]
    items: list[ParsedItem]


class RubricConfigDocumentSchema:
    """Base interface for schema-specific rubric config parsing."""

    version: Optional[int] = None

    def parse_document(
        self,
        document: Any,
        *,
        path: Path,
        doc_index: int,
        strict: bool,
    ) -> RubricConfigDocumentResult:
        raise NotImplementedError


class LegacyRubricConfigSchema(RubricConfigDocumentSchema):
    """Schema handling documents without an explicit version."""

    version = None

    def parse_document(
        self,
        document: Any,
        *,
        path: Path,
        doc_index: int,
        strict: bool,
    ) -> RubricConfigDocumentResult:
        defaults = _extract_config_defaults(document, path, doc_index)
        entries = _extract_rubric_items(document, context=None, doc_index=doc_index)
        return _build_document_configs(entries, defaults, path=path, doc_index=doc_index, strict=strict)


class Version1RubricConfigSchema(LegacyRubricConfigSchema):
    """Schema for version 1 documents (currently aligned with legacy layout)."""

    version = 1


class RubricConfigParser:
    """Parses rubric configuration files and produces typed suites."""

    def __init__(self, *, schemas: Optional[dict[Optional[int], RubricConfigDocumentSchema]] = None):
        self._schemas = schemas or {
            None: LegacyRubricConfigSchema(),
            1: Version1RubricConfigSchema(),
        }
        if None not in self._schemas:
            raise ValueError("At least one default schema (key=None) must be provided.")

    def parse(self, path: Path, *, strict: bool = True) -> tuple[RubricSuite, list[ParsedItem]]:
        documents = _load_yaml_documents(path)
        configs: dict[str, RubricConfig] = {}
        parsed_items: list[ParsedItem] = []
        detected_version: Optional[int] = None
        document_indices = []

        for doc_index, document in enumerate(documents):
            if document:
                document_indices.append(doc_index)
            if not document:
                continue

            doc_version = self._coerce_optional_version(document, path, doc_index)
            if doc_version is not None:
                if detected_version is None:
                    detected_version = doc_version
                elif detected_version != doc_version:
                    raise CLIError(
                        f"Rubric config '{path}' mixes different version numbers across documents."
                    )

        schema = self._select_schema(detected_version)

        for doc_index in document_indices:
            document = documents[doc_index]
            if not document:
                continue

            result = schema.parse_document(
                document,
                path=path,
                doc_index=doc_index,
                strict=strict,
            )
            parsed_items.extend(result.items)
            for rubric_id, config in result.configs.items():
                if rubric_id in configs:
                    raise CLIError(f"Duplicate rubric id '{rubric_id}' detected in '{path}'.")
                configs[rubric_id] = config

        if strict and not configs:
            raise CLIError(f"No rubric entries found in '{path}'.")

        suite = RubricSuite(source_path=path, version=detected_version, configs=configs)
        return suite, parsed_items

    def _select_schema(self, version: Optional[int]) -> RubricConfigDocumentSchema:
        if version in self._schemas:
            return self._schemas[version]
        if version is None:
            return self._schemas[None]
        raise CLIError(f"Unsupported rubric config version '{version}'.")

    @staticmethod
    def _coerce_optional_version(document: Any, path: Path, doc_index: int) -> Optional[int]:
        if not isinstance(document, dict):
            return None
        version_value = document.get("version")
        if version_value is None:
            return None
        if isinstance(version_value, int):
            if version_value < 0:
                raise CLIError(
                    f"Version number in '{path}' document {doc_index} must be non-negative."
                )
            return version_value
        raise CLIError(
            f"Version field in '{path}' document {doc_index} must be an integer."
        )


def _build_document_configs(
    entries: Sequence[ParsedItem],
    defaults: dict[str, Any],
    *,
    path: Path,
    doc_index: int,
    strict: bool,
) -> RubricConfigDocumentResult:
    configs: dict[str, RubricConfig] = {}
    parsed_items: list[ParsedItem] = []

    for item in entries:
        payload = item.payload
        parsed_items.append(ParsedItem(label=item.label, payload=payload))
        if not isinstance(payload, dict):
            continue
        if "extra_info" in payload:
            message = (
                f"Rubric entry in '{path}' (document {doc_index + 1}) must not include 'extra_info'."
            )
            if strict:
                raise CLIError(message)
            continue

        rubric_key_raw = payload.get("id")
        if not isinstance(rubric_key_raw, str) or not rubric_key_raw.strip():
            if strict:
                raise CLIError(
                    f"Rubric entry in '{path}' (document {doc_index}) is missing a non-empty 'id'."
                )
            continue
        rubric_key = rubric_key_raw.strip()
        if rubric_key in configs:
            raise CLIError(f"Duplicate rubric id '{rubric_key}' detected in '{path}'.")

        rubric_text = payload.get("rubric")
        if not isinstance(rubric_text, str) or not rubric_text.strip():
            if strict:
                raise CLIError(
                    f"Rubric '{rubric_key}' in '{path}' must include a non-empty 'rubric' string."
                )
            continue

        model_info = payload.get("model_info", defaults.get("model_info"))
        if not isinstance(model_info, dict):
            if strict:
                raise CLIError(
                    f"Rubric '{rubric_key}' in '{path}' must include a 'model_info' mapping."
                )
            continue

        try:
            score_min = coerce_optional_float(
                payload.get("score_min", defaults.get("score_min")),
                "score_min",
                f"rubric '{rubric_key}' in {path}",
            )
            score_max = coerce_optional_float(
                payload.get("score_max", defaults.get("score_max")),
                "score_max",
                f"rubric '{rubric_key}' in {path}",
            )
        except CLIError:
            if strict:
                raise
            continue

        system_prompt = payload.get("system_prompt", defaults.get("system_prompt"))

        original_input = payload.get("original_input", defaults.get("original_input"))
        if not isinstance(original_input, str):
            original_input = None

        ground_truth = payload.get("ground_truth", defaults.get("ground_truth"))

        label = item.label or f"document[{doc_index}]"
        source_label = f"{path}:{label}"

        configs[rubric_key] = RubricConfig(
            rubric_id=rubric_key,
            rubric_text=rubric_text,
            model_info=copy.deepcopy(model_info),
            score_min=score_min,
            score_max=score_max,
            system_prompt=system_prompt if isinstance(system_prompt, str) else None,
            original_input=original_input,
            ground_truth=ground_truth if isinstance(ground_truth, str) else None,
            source_label=source_label,
        )

    return RubricConfigDocumentResult(configs=configs, items=parsed_items)


def discover_rubric_config_path(config_arg: Optional[str], data_path: Path) -> Path:
    if config_arg:
        candidate = Path(config_arg).expanduser()
        if not candidate.exists():
            raise CLIError(f"Rubric config path '{candidate}' does not exist.")
        if candidate.is_dir():
            raise CLIError(f"Rubric config path '{candidate}' is a directory.")
        return candidate

    candidates: list[Path] = []
    candidates.append(data_path.parent / "rubric_configs.yaml")
    candidates.append(Path.cwd() / "rubric_configs.yaml")
    candidates.append(Path.cwd() / "examples" / "rubric_configs.yaml")

    checked: list[Path] = []
    for candidate in dict.fromkeys(candidates):
        checked.append(candidate)
        if candidate.exists() and candidate.is_file():
            return candidate

    searched = ", ".join(str(path) for path in checked)
    raise CLIError(
        "Unable to locate a rubric config file. Provide --config explicitly. "
        f"Paths checked: {searched}"
    )


def load_rubric_configs(path: Path) -> list[ParsedItem]:
    parser = RubricConfigParser()
    _, items = parser.parse(path, strict=False)
    return items


def load_rubric_suite(path: Path) -> RubricSuite:
    parser = RubricConfigParser()
    suite, _ = parser.parse(path)
    return suite


def render_yaml_items(items: Sequence[ParsedItem], label: str) -> str:
    blocks: list[str] = []
    total = len(items)

    for index, item in enumerate(items, start=1):
        header = f"{label} #{index}"
        if item.label:
            header += f" ({item.label})"
        dumped = yaml.dump(
            item.payload,
            Dumper=_LiteralSafeDumper,
            sort_keys=False,
            indent=2,
            allow_unicode=True,
        ).rstrip()

        snippet = [header, dumped]
        if index != total:
            snippet.append("")
        blocks.append("\n".join(snippet))

    return "\n".join(blocks)


def _load_yaml_documents(path: Path) -> list[Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return list(yaml.safe_load_all(fh))
    except yaml.YAMLError as exc:
        raise CLIError(f"Failed to parse YAML in '{path}': {exc}") from exc
    except OSError as exc:
        raise CLIError(f"Unable to read rubric config '{path}': {exc}") from exc


def _extract_config_defaults(document: Any, path: Path, doc_index: int) -> dict[str, Any]:
    if not isinstance(document, dict):
        return {
            "model_info": None,
            "score_min": None,
            "score_max": None,
            "system_prompt": None,
            "original_input": None,
            "ground_truth": None,
        }

    source = f"document[{doc_index}] in {path}"

    defaults: dict[str, Any] = {}
    if "default_extra_info" in document:
        raise CLIError(
            f"Rubric config document {doc_index + 1} in {path} must not include 'default_extra_info'; extra_info is no longer supported."
        )
    defaults["model_info"] = document.get("default_model_info")
    defaults["score_min"] = coerce_optional_float(
        document.get("default_score_min"), "default_score_min", source
    )
    defaults["score_max"] = coerce_optional_float(
        document.get("default_score_max"), "default_score_max", source
    )
    defaults["system_prompt"] = document.get("default_system_prompt")
    defaults["original_input"] = document.get("default_original_input")
    defaults["ground_truth"] = document.get("default_ground_truth")
    return defaults


def _extract_rubric_items(node: Any, context: Optional[str], doc_index: int) -> list[ParsedItem]:
    items: list[ParsedItem] = []

    if node is None:
        return items

    if isinstance(node, dict):
        if "rubric" in node and isinstance(node["rubric"], str):
            label = context or f"document[{doc_index}]"
            items.append(ParsedItem(label=label, payload=node))
        else:
            for key, value in node.items():
                next_context = str(key) if isinstance(key, str) else context
                items.extend(_extract_rubric_items(value, context=next_context, doc_index=doc_index))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            idx_context = f"{context}[{index}]" if context else None
            items.extend(_extract_rubric_items(value, context=idx_context, doc_index=doc_index))

    return items


class _LiteralSafeDumper(yaml.SafeDumper):
    """YAML dumper that preserves multiline strings with literal blocks."""


def _represent_str(dumper: yaml.Dumper, data: str):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return SafeRepresenter.represent_str(dumper, data)


_LiteralSafeDumper.add_representer(str, _represent_str)
