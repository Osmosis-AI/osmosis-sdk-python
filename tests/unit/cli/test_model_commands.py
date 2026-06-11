"""Tests for osmosis_ai.cli.commands.model."""

from __future__ import annotations

import json
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

import osmosis_ai.platform.cli.model as platform_model_module
import osmosis_ai.platform.cli.utils as utils_module
from osmosis_ai.cli.console import Console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import (
    DetailResult,
    ListResult,
    OperationResult,
    OutputFormat,
    SectionedListResult,
    override_output_context,
)
from osmosis_ai.platform.api.models import (
    BaseModelInfo,
    LoraModelDetail,
    LoraModelInfo,
    LoraModelSummary,
    PaginatedBaseModels,
    PaginatedLoraModels,
)
from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

AUTH_CREDENTIALS = object()
GIT_IDENTITY = "acme/rollouts"
REPO_URL = "https://github.com/acme/rollouts.git"
PROJECT_ROOT = Path("/repo")


def assert_git_context(data: dict[str, object]) -> None:
    assert data["workspace_directory"] == "/repo"
    assert data["git"] == {
        "identity": GIT_IDENTITY,
        "remote_url": REPO_URL,
    }
    assert "workspace" not in data


def _lora_model_summary(name: str, status: str = "active") -> LoraModelSummary:
    return LoraModelSummary(id="lora_1", model_name=name, status=status)


def _lora_model_detail(
    name: str,
    deployment_status: str | None = None,
    *,
    uploaded_by: str | None = "Ada Lovelace",
    has_deployment_info: bool = True,
) -> LoraModelDetail:
    return LoraModelDetail(
        id="lora_1",
        model_name=name,
        base_model="Qwen/Qwen3-8B",
        training_run_name="run1",
        checkpoint_step=100,
        reward=0.85,
        deployment_status=deployment_status,
        created_at="2026-06-01T00:00:00Z",
        hf_upload_status="uploaded",
        hf_url="https://huggingface.co/acme/qwen3-run1-step-100",
        uploaded_by=uploaded_by,
        has_deployment_info=has_deployment_info,
        inference_model=f"Qwen/Qwen3-8B:{name}",
        platform_url="https://platform.osmosis.ai/acme/models/lora_1",
    )


@pytest.fixture()
def console_capture(monkeypatch: pytest.MonkeyPatch) -> StringIO:
    output = StringIO()
    console = Console(file=output, force_terminal=False)
    monkeypatch.setattr(platform_model_module, "console", console)
    monkeypatch.setattr(utils_module, "console", console)
    return output


@pytest.fixture()
def mock_git_context(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    context = SimpleNamespace(
        workspace_directory=PROJECT_ROOT,
        git_identity=GIT_IDENTITY,
        repo_url=REPO_URL,
        credentials=AUTH_CREDENTIALS,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.model.require_git_workspace_directory_context",
        lambda: context,
    )
    return context


def test_model_list_requires_linked_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from osmosis_ai.cli.main import main

    monkeypatch.chdir(tmp_path)

    rc = main(["--json", "model", "list"])

    assert rc == 1
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "WORKSPACE_REQUIRED"
    assert "Osmosis workspace directory" in error["message"]


@pytest.mark.parametrize("limit", ["0", "51"])
def test_model_list_rejects_out_of_range_limit(limit: str, capsys) -> None:
    from osmosis_ai.cli.main import main

    rc = main(["model", "list", "--limit", limit])

    assert rc == 2
    assert "is not in the range 1<=x<=50" in capsys.readouterr().err


@pytest.mark.parametrize("limit", ["0", "51"])
def test_dataset_list_rejects_out_of_range_limit(limit: str, capsys) -> None:
    from osmosis_ai.cli.main import main

    rc = main(["dataset", "list", "--limit", limit])

    assert rc == 2
    assert "is not in the range 1<=x<=50" in capsys.readouterr().err


def _base_model(**overrides: object) -> BaseModelInfo:
    defaults: dict = {
        "id": "model_1",
        "model_name": "Qwen/Qwen3",
        "base_model": "Qwen/Qwen3",
        "creator_name": "brian",
        "created_at": "2026-04-20T00:00:00Z",
    }
    defaults.update(overrides)
    return BaseModelInfo(**defaults)


def _lora_model(**overrides: object) -> LoraModelInfo:
    defaults: dict = {
        "id": "lora_1",
        "model_name": "qwen3-run1-step-100",
        "base_model": "Qwen/Qwen3",
        "training_run_name": "qwen3-run1",
        "checkpoint_step": 100,
        "reward": 0.85,
        "deployment_status": "active",
        "created_at": "2026-04-21T00:00:00Z",
    }
    defaults.update(overrides)
    return LoraModelInfo(**defaults)


def _fake_list_client(
    base_models: list[BaseModelInfo],
    lora_models: list[LoraModelInfo],
    captured: dict[str, object] | None = None,
    *,
    base_has_more: bool = False,
    lora_has_more: bool = False,
    base_next_offset: int | None = None,
    lora_next_offset: int | None = None,
    active_deployments: int = 0,
    max_active_deployments: int = 0,
):
    class FakeClient:
        def list_base_models(
            self, limit=30, offset=0, *, git_identity, credentials=None
        ):
            assert credentials is AUTH_CREDENTIALS
            assert git_identity == GIT_IDENTITY
            if captured is not None:
                captured["base"] = {"limit": limit, "offset": offset}
            return PaginatedBaseModels(
                models=base_models,
                total_count=len(base_models),
                has_more=base_has_more,
                next_offset=base_next_offset,
            )

        def list_lora_models(
            self, limit=30, offset=0, *, git_identity, credentials=None
        ):
            assert credentials is AUTH_CREDENTIALS
            assert git_identity == GIT_IDENTITY
            if captured is not None:
                captured["lora"] = {"limit": limit, "offset": offset}
            return PaginatedLoraModels(
                models=lora_models,
                total_count=len(lora_models),
                has_more=lora_has_more,
                next_offset=lora_next_offset,
                active_deployments=active_deployments,
                max_active_deployments=max_active_deployments,
            )

    return FakeClient


_BASE_COLUMN_KEYS = ["model_name", "created_at", "creator_name"]
_BASE_COLUMN_LABELS = ["Name", "Created", "Created By"]
_LORA_COLUMN_KEYS = [
    "model_name",
    "base_model",
    "training_run_name",
    "checkpoint_step",
    "reward",
    "created_at",
    "deployment_status",
]
_LORA_COLUMN_LABELS = [
    "Name",
    "Base Model",
    "Training Run",
    "Checkpoint Step",
    "Training Reward",
    "Created",
    "Deployment Status",
]


@pytest.mark.usefixtures("mock_git_context")
class TestListModels:
    def test_empty_list(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        monkeypatch.setattr(
            platform_model_module, "OsmosisClient", _fake_list_client([], [])
        )
        result = platform_model_module.list_models(limit=30, all_=False)

        assert isinstance(result, SectionedListResult)
        assert [section.key for section in result.sections] == [
            "base_models",
            "lora_models",
        ]
        for section in result.sections:
            assert section.items == []
            assert section.total_count == 0
            assert section.has_more is False
            assert section.next_offset is None
        assert result.display_hints == []
        assert_git_context(result.extra)

    def test_list_sections_base_models_before_lora_models(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        captured: dict[str, object] = {}
        monkeypatch.setattr(
            platform_model_module,
            "OsmosisClient",
            _fake_list_client([_base_model()], [_lora_model()], captured),
        )
        result = platform_model_module.list_models(limit=10, all_=False)

        assert captured == {
            "base": {"limit": 10, "offset": 0},
            "lora": {"limit": 10, "offset": 0},
        }
        assert isinstance(result, SectionedListResult)
        base_section, lora_section = result.sections
        assert (base_section.key, base_section.title) == ("base_models", "Base Models")
        assert (lora_section.key, lora_section.title) == ("lora_models", "LoRA Models")
        assert [i["model_name"] for i in base_section.items] == ["Qwen/Qwen3"]
        assert [i["model_name"] for i in lora_section.items] == ["qwen3-run1-step-100"]
        assert "status" not in base_section.items[0]
        assert lora_section.items[0]["deployment_status"] == "active"
        assert lora_section.items[0]["checkpoint_step"] == 100
        assert lora_section.items[0]["reward"] == 0.85
        assert lora_section.items[0]["training_run_name"] == "qwen3-run1"
        assert lora_section.display_items is not None
        assert lora_section.display_items[0]["reward"] == "0.85"
        assert base_section.total_count == 1
        assert lora_section.total_count == 1
        assert result.display_hints == [
            "Deploy a LoRA model with: osmosis model deploy <name>"
        ]

    def test_list_items_carry_no_type_discriminator(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        monkeypatch.setattr(
            platform_model_module,
            "OsmosisClient",
            _fake_list_client([_base_model()], [_lora_model()]),
        )
        result = platform_model_module.list_models(limit=30, all_=False)

        assert isinstance(result, SectionedListResult)
        for section in result.sections:
            assert all("type" not in item for item in section.items)
            assert section.display_items is not None
            assert all("type" not in item for item in section.display_items)

    def test_list_sections_use_per_type_columns(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        monkeypatch.setattr(
            platform_model_module,
            "OsmosisClient",
            _fake_list_client([_base_model()], [_lora_model()]),
        )
        result = platform_model_module.list_models(limit=30, all_=False)

        assert isinstance(result, SectionedListResult)
        base_section, lora_section = result.sections
        assert [c.key for c in base_section.columns] == _BASE_COLUMN_KEYS
        assert [c.label for c in base_section.columns] == _BASE_COLUMN_LABELS
        assert [c.key for c in lora_section.columns] == _LORA_COLUMN_KEYS
        assert [c.label for c in lora_section.columns] == _LORA_COLUMN_LABELS

    def test_list_display_items_fill_missing_fields_with_dashes(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        monkeypatch.setattr(
            platform_model_module,
            "OsmosisClient",
            _fake_list_client(
                [_base_model(base_model=None, creator_name=None)],
                [
                    _lora_model(
                        base_model=None,
                        training_run_name=None,
                        checkpoint_step=None,
                        reward=None,
                        deployment_status=None,
                    )
                ],
            ),
        )
        result = platform_model_module.list_models(limit=30, all_=False)

        assert isinstance(result, SectionedListResult)
        base_section, lora_section = result.sections
        assert base_section.display_items is not None
        assert lora_section.display_items is not None
        (base_display,) = base_section.display_items
        (lora_display,) = lora_section.display_items
        assert base_display["creator_name"] == "—"
        assert lora_display["deployment_status"] == "—"
        assert lora_display["base_model"] == "—"
        assert lora_display["training_run_name"] == "—"
        assert lora_display["checkpoint_step"] == "—"
        assert lora_display["reward"] == "—"
        # Raw JSON items keep nulls — display dashes must not leak into them.
        assert lora_section.items[0]["deployment_status"] is None
        assert lora_section.items[0]["checkpoint_step"] is None
        assert lora_section.items[0]["reward"] is None

    def test_list_sections_pass_cursors_through_verbatim(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        monkeypatch.setattr(
            platform_model_module,
            "OsmosisClient",
            _fake_list_client(
                [_base_model()],
                [_lora_model()],
                base_has_more=True,
                base_next_offset=1,
                lora_has_more=False,
                lora_next_offset=2,
            ),
        )
        result = platform_model_module.list_models(limit=30, all_=False)

        assert isinstance(result, SectionedListResult)
        base_section, lora_section = result.sections
        assert base_section.has_more is True
        assert base_section.next_offset == 1
        assert lora_section.has_more is False
        assert lora_section.next_offset == 2

    def test_list_all_drains_every_page_per_section(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        base_pages = {
            0: PaginatedBaseModels(
                models=[_base_model(id="model_1", model_name="base-a")],
                total_count=2,
                has_more=True,
                next_offset=1,
            ),
            1: PaginatedBaseModels(
                models=[_base_model(id="model_2", model_name="base-b")],
                total_count=2,
                has_more=False,
                next_offset=None,
            ),
        }
        lora_pages = {
            0: PaginatedLoraModels(
                models=[_lora_model(id="lora_1", model_name="lora-a")],
                total_count=2,
                has_more=True,
                next_offset=1,
            ),
            1: PaginatedLoraModels(
                models=[_lora_model(id="lora_2", model_name="lora-b")],
                total_count=2,
                has_more=False,
                next_offset=None,
            ),
        }
        calls: dict[str, list[dict[str, int]]] = {"base": [], "lora": []}

        class FakeClient:
            def list_base_models(
                self, limit=30, offset=0, *, git_identity, credentials=None
            ):
                assert credentials is AUTH_CREDENTIALS
                assert git_identity == GIT_IDENTITY
                calls["base"].append({"limit": limit, "offset": offset})
                return base_pages[offset]

            def list_lora_models(
                self, limit=30, offset=0, *, git_identity, credentials=None
            ):
                assert credentials is AUTH_CREDENTIALS
                assert git_identity == GIT_IDENTITY
                calls["lora"].append({"limit": limit, "offset": offset})
                return lora_pages[offset]

        monkeypatch.setattr(platform_model_module, "OsmosisClient", FakeClient)
        result = platform_model_module.list_models(limit=DEFAULT_PAGE_SIZE, all_=True)

        expected_calls = [
            {"limit": DEFAULT_PAGE_SIZE, "offset": 0},
            {"limit": DEFAULT_PAGE_SIZE, "offset": 1},
        ]
        assert calls == {"base": expected_calls, "lora": expected_calls}
        assert isinstance(result, SectionedListResult)
        base_section, lora_section = result.sections
        assert [i["model_name"] for i in base_section.items] == ["base-a", "base-b"]
        assert [i["model_name"] for i in lora_section.items] == ["lora-a", "lora-b"]
        for section in result.sections:
            assert section.total_count == 2
            assert section.has_more is False
            assert section.next_offset is None

    def test_list_type_base_only_calls_base_endpoint(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        captured: dict[str, object] = {}
        monkeypatch.setattr(
            platform_model_module,
            "OsmosisClient",
            _fake_list_client(
                [_base_model()], [_lora_model()], captured, base_next_offset=None
            ),
        )
        result = platform_model_module.list_models(limit=30, all_=False, type_="base")

        assert list(captured) == ["base"]
        assert isinstance(result, ListResult)
        assert result.title == "Base Models"
        assert [i["model_name"] for i in result.items] == ["Qwen/Qwen3"]
        assert all("type" not in item for item in result.items)
        assert [c.key for c in result.columns] == _BASE_COLUMN_KEYS
        assert result.total_count == 1
        assert result.next_offset is None
        assert result.display_hints == []
        assert_git_context(result.extra)

    def test_list_type_lora_only_calls_lora_endpoint(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        captured: dict[str, object] = {}
        monkeypatch.setattr(
            platform_model_module,
            "OsmosisClient",
            _fake_list_client(
                [_base_model()], [_lora_model()], captured, lora_next_offset=1
            ),
        )
        result = platform_model_module.list_models(limit=30, all_=False, type_="lora")

        assert list(captured) == ["lora"]
        assert isinstance(result, ListResult)
        assert result.title == "LoRA Models"
        assert [i["model_name"] for i in result.items] == ["qwen3-run1-step-100"]
        assert all("type" not in item for item in result.items)
        assert [c.key for c in result.columns] == _LORA_COLUMN_KEYS
        assert result.next_offset == 1
        assert result.display_hints == [
            "Deploy a LoRA model with: osmosis model deploy <name>"
        ]
        assert_git_context(result.extra)

    def test_list_rejects_invalid_type(self) -> None:
        with pytest.raises(CLIError, match=r"Type must be 'all', 'base', or 'lora'\."):
            platform_model_module.list_models(limit=30, all_=False, type_="bogus")

    def test_list_items_carry_deploy_metadata(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        monkeypatch.setattr(
            platform_model_module,
            "OsmosisClient",
            _fake_list_client(
                [],
                [
                    _lora_model(
                        deployed_at="2026-04-22T00:00:00Z", deployed_by="brian"
                    ),
                    _lora_model(id="lora_2", model_name="undeployed"),
                ],
            ),
        )
        result = platform_model_module.list_models(limit=30, all_=False, type_="lora")

        assert isinstance(result, ListResult)
        assert result.items[0]["deployed_at"] == "2026-04-22T00:00:00Z"
        assert result.items[0]["deployed_by"] == "brian"
        assert result.items[1]["deployed_at"] is None
        assert result.items[1]["deployed_by"] is None
        # No table column for deploy metadata — JSON contract only.
        assert "deployed_at" not in [c.key for c in result.columns]
        assert "deployed_by" not in [c.key for c in result.columns]

    def test_list_shows_deployment_quota_hint_under_lora_table(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        monkeypatch.setattr(
            platform_model_module,
            "OsmosisClient",
            _fake_list_client(
                [_base_model()],
                [_lora_model()],
                active_deployments=2,
                max_active_deployments=5,
            ),
        )
        result = platform_model_module.list_models(limit=30, all_=False)

        assert isinstance(result, SectionedListResult)
        assert result.display_hints == [
            "2 of 5 inference deployments used",
            "Deploy a LoRA model with: osmosis model deploy <name>",
        ]

    def test_list_type_lora_shows_deployment_quota_hint(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        monkeypatch.setattr(
            platform_model_module,
            "OsmosisClient",
            _fake_list_client(
                [],
                [],
                active_deployments=0,
                max_active_deployments=5,
            ),
        )
        result = platform_model_module.list_models(limit=30, all_=False, type_="lora")

        assert isinstance(result, ListResult)
        assert result.display_hints == ["0 of 5 inference deployments used"]

    def test_list_omits_quota_hint_when_server_reports_no_quota(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        monkeypatch.setattr(
            platform_model_module,
            "OsmosisClient",
            _fake_list_client([], [_lora_model()]),
        )
        result = platform_model_module.list_models(limit=30, all_=False)

        assert isinstance(result, SectionedListResult)
        assert result.display_hints == [
            "Deploy a LoRA model with: osmosis model deploy <name>"
        ]

    def test_list_all_captures_quota_from_first_page(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        monkeypatch.setattr(
            platform_model_module,
            "OsmosisClient",
            _fake_list_client(
                [],
                [_lora_model()],
                active_deployments=1,
                max_active_deployments=5,
            ),
        )
        result = platform_model_module.list_models(
            limit=DEFAULT_PAGE_SIZE, all_=True, type_="lora"
        )

        assert isinstance(result, ListResult)
        assert result.extra["active_deployments"] == 1
        assert result.extra["max_active_deployments"] == 5


def _render_to_json_envelope(result) -> dict[str, object]:
    import io
    from contextlib import redirect_stderr, redirect_stdout

    from osmosis_ai.cli.output.renderer import render

    out, err = io.StringIO(), io.StringIO()
    with override_output_context(format=OutputFormat.json) as ctx:
        with redirect_stdout(out), redirect_stderr(err):
            render(result, ctx)
    assert err.getvalue() == ""
    return json.loads(out.getvalue())


@pytest.mark.usefixtures("mock_git_context")
class TestListModelsJsonEnvelope:
    def test_type_all_envelope_carries_deployment_quota(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        monkeypatch.setattr(
            platform_model_module,
            "OsmosisClient",
            _fake_list_client(
                [_base_model()],
                [_lora_model()],
                active_deployments=2,
                max_active_deployments=5,
            ),
        )
        result = platform_model_module.list_models(limit=30, all_=False)

        envelope = _render_to_json_envelope(result)
        assert envelope["active_deployments"] == 2
        assert envelope["max_active_deployments"] == 5
        assert envelope["base_models"]["total_count"] == 1
        assert envelope["lora_models"]["items"][0]["deployed_at"] is None
        assert envelope["lora_models"]["items"][0]["deployed_by"] is None

    def test_type_lora_envelope_carries_deployment_quota(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        monkeypatch.setattr(
            platform_model_module,
            "OsmosisClient",
            _fake_list_client(
                [],
                [_lora_model()],
                active_deployments=2,
                max_active_deployments=5,
            ),
        )
        result = platform_model_module.list_models(limit=30, all_=False, type_="lora")

        envelope = _render_to_json_envelope(result)
        assert envelope["active_deployments"] == 2
        assert envelope["max_active_deployments"] == 5

    def test_type_base_envelope_omits_deployment_quota(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        monkeypatch.setattr(
            platform_model_module,
            "OsmosisClient",
            _fake_list_client(
                [_base_model()],
                [_lora_model()],
                active_deployments=2,
                max_active_deployments=5,
            ),
        )
        result = platform_model_module.list_models(limit=30, all_=False, type_="base")

        envelope = _render_to_json_envelope(result)
        assert "active_deployments" not in envelope
        assert "max_active_deployments" not in envelope


@pytest.mark.usefixtures("mock_git_context")
class TestInfo:
    def test_info_accepts_lora_model_name(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        captured: dict[str, object] = {}

        class FakeClient:
            def get_lora_model(
                self, lora_model_name, *, git_identity, credentials=None
            ):
                captured["lora_model_name"] = lora_model_name
                captured["credentials"] = credentials
                captured["git_identity"] = git_identity
                return _lora_model_detail("qwen3-run1-step-100")

        monkeypatch.setattr(platform_model_module, "OsmosisClient", FakeClient)
        result = platform_model_module.info("qwen3-run1-step-100")
        assert captured == {
            "lora_model_name": "qwen3-run1-step-100",
            "credentials": AUTH_CREDENTIALS,
            "git_identity": GIT_IDENTITY,
        }
        assert isinstance(result, DetailResult)
        assert result.title == "LoRA Model Info"
        lora_model = result.data["lora_model"]
        assert lora_model["model_name"] == "qwen3-run1-step-100"
        assert lora_model["base_model"] == "Qwen/Qwen3-8B"
        assert lora_model["hf_upload_status"] == "uploaded"
        assert lora_model["hf_url"] == "https://huggingface.co/acme/qwen3-run1-step-100"
        assert lora_model["uploaded_by"] == "Ada Lovelace"
        assert lora_model["deployment_status"] is None
        assert (
            result.data["platform_url"]
            == "https://platform.osmosis.ai/acme/models/lora_1"
        )
        assert_git_context(result.data)
        labels = [field.label for field in result.fields]
        assert labels == [
            "Name",
            "Base Model",
            "Training Run",
            "Checkpoint Step",
            "Training Reward",
            "Created",
            "HF Upload Status",
            "Hugging Face",
            "HF Uploaded By",
            "Deployment Status",
        ]

    def test_info_without_deployment_info_hides_rows_and_hints(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        class FakeClient:
            def get_lora_model(
                self, lora_model_name, *, git_identity, credentials=None
            ):
                return _lora_model_detail(
                    lora_model_name, uploaded_by=None, has_deployment_info=False
                )

        monkeypatch.setattr(platform_model_module, "OsmosisClient", FakeClient)
        result = platform_model_module.info("qwen3-run1-step-100")

        labels = [field.label for field in result.fields]
        assert "Deployment Status" not in labels
        assert "Deployed By" not in labels
        assert "HF Uploaded By" not in labels
        assert not any("osmosis model deploy" in hint for hint in result.display_hints)
        assert not any(
            "osmosis model undeploy" in hint for hint in result.display_hints
        )
        lora_model = result.data["lora_model"]
        assert "deployment_status" not in lora_model
        assert "deployed_at" not in lora_model
        assert "deployed_by" not in lora_model

    def test_info_always_includes_platform_url_key(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        class FakeClient:
            def get_lora_model(
                self, lora_model_name, *, git_identity, credentials=None
            ):
                detail = _lora_model_detail(lora_model_name)
                detail.platform_url = None
                return detail

        monkeypatch.setattr(platform_model_module, "OsmosisClient", FakeClient)
        result = platform_model_module.info("qwen3-run1-step-100")

        assert "platform_url" in result.data
        assert result.data["platform_url"] is None
        assert not any(hint.startswith("View:") for hint in result.display_hints)

    def test_info_hints_follow_deployment_status(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        details = {"status": None}

        class FakeClient:
            def get_lora_model(
                self, lora_model_name, *, git_identity, credentials=None
            ):
                return _lora_model_detail(
                    lora_model_name, deployment_status=details["status"]
                )

        monkeypatch.setattr(platform_model_module, "OsmosisClient", FakeClient)

        result = platform_model_module.info("qwen3-run1-step-100")
        assert (
            "Deploy with: osmosis model deploy qwen3-run1-step-100"
            in result.display_hints
        )

        details["status"] = "active"
        result = platform_model_module.info("qwen3-run1-step-100")
        assert (
            "Undeploy with: osmosis model undeploy qwen3-run1-step-100"
            in result.display_hints
        )
        query_hint = next(
            hint for hint in result.display_hints if hint.startswith("Query it")
        )
        assert '"Authorization: Bearer $OSMOSIS_API_KEY"' in query_hint
        assert '"model": "Qwen/Qwen3-8B:qwen3-run1-step-100"' in query_hint
        assert (
            "Create an API key: https://platform.osmosis.ai/acme/api-keys"
            in result.display_hints
        )

    def test_info_escapes_name_in_spinner(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        captured: dict[str, object] = {}

        @contextmanager
        def record_spinner(message):
            captured["message"] = message
            yield

        class FakeClient:
            def get_lora_model(
                self, lora_model_name, *, git_identity, credentials=None
            ):
                captured["lora_model_name"] = lora_model_name
                return _lora_model_detail("qwen3-run1-step-100")

        monkeypatch.setattr(platform_model_module, "OsmosisClient", FakeClient)

        with override_output_context(
            format=OutputFormat.rich, interactive=True
        ) as output:
            monkeypatch.setattr(output, "status", record_spinner)
            result = platform_model_module.info("[red]bad[/red]")

        assert captured["lora_model_name"] == "[red]bad[/red]"
        assert captured["message"] == 'Fetching LoRA model "\\[red]bad\\[/red]"...'
        assert isinstance(result, DetailResult)


@pytest.mark.usefixtures("mock_git_context")
class TestDeploy:
    def test_deploy_accepts_lora_model_name(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        captured: dict[str, object] = {}

        class FakeClient:
            def deploy_lora_model(
                self, lora_model_name, *, git_identity, credentials=None
            ):
                captured["lora_model_name"] = lora_model_name
                captured["credentials"] = credentials
                captured["git_identity"] = git_identity
                return _lora_model_summary("qwen3-run1-step-100")

        monkeypatch.setattr(platform_model_module, "OsmosisClient", FakeClient)
        result = platform_model_module.deploy("qwen3-run1-step-100")
        assert captured == {
            "lora_model_name": "qwen3-run1-step-100",
            "credentials": AUTH_CREDENTIALS,
            "git_identity": GIT_IDENTITY,
        }
        assert isinstance(result, OperationResult)
        assert result.operation == "model.deploy"
        assert result.status == "success"
        assert result.resource == {
            "id": "lora_1",
            "model_name": "qwen3-run1-step-100",
            "status": "active",
            "git": {"identity": GIT_IDENTITY, "remote_url": REPO_URL},
            "workspace_directory": "/repo",
        }
        assert result.message == "LoRA model deployed: qwen3-run1-step-100"

    def test_deploy_escapes_name_in_spinner(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        captured: dict[str, object] = {}

        @contextmanager
        def record_spinner(message):
            captured["message"] = message
            yield

        class FakeClient:
            def deploy_lora_model(
                self, lora_model_name, *, git_identity, credentials=None
            ):
                captured["lora_model_name"] = lora_model_name
                captured["git_identity"] = git_identity
                return _lora_model_summary("qwen3-run1-step-100")

        monkeypatch.setattr(platform_model_module, "OsmosisClient", FakeClient)

        with override_output_context(
            format=OutputFormat.rich, interactive=True
        ) as output:
            monkeypatch.setattr(output, "status", record_spinner)
            result = platform_model_module.deploy("[red]bad[/red]")

        assert captured["lora_model_name"] == "[red]bad[/red]"
        assert captured["git_identity"] == GIT_IDENTITY
        assert captured["message"] == 'Deploying LoRA model "\\[red]bad\\[/red]"...'
        assert isinstance(result, OperationResult)


@pytest.mark.usefixtures("mock_git_context")
class TestUndeploy:
    def test_undeploy_accepts_lora_model_name(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        captured: dict[str, object] = {}

        class FakeClient:
            def undeploy_lora_model(
                self, lora_model_name, *, git_identity, credentials=None
            ):
                captured["lora_model_name"] = lora_model_name
                captured["credentials"] = credentials
                captured["git_identity"] = git_identity
                return _lora_model_summary("qwen3-run1-step-100", status="inactive")

        monkeypatch.setattr(platform_model_module, "OsmosisClient", FakeClient)
        result = platform_model_module.undeploy("qwen3-run1-step-100")
        assert captured == {
            "lora_model_name": "qwen3-run1-step-100",
            "credentials": AUTH_CREDENTIALS,
            "git_identity": GIT_IDENTITY,
        }
        assert isinstance(result, OperationResult)
        assert result.operation == "model.undeploy"
        assert result.status == "success"
        assert result.resource == {
            "id": "lora_1",
            "model_name": "qwen3-run1-step-100",
            "status": "inactive",
            "git": {"identity": GIT_IDENTITY, "remote_url": REPO_URL},
            "workspace_directory": "/repo",
        }
        assert result.message == "LoRA model undeployed: qwen3-run1-step-100"

    def test_undeploy_escapes_name_in_spinner(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        captured: dict[str, object] = {}

        @contextmanager
        def record_spinner(message):
            captured["message"] = message
            yield

        class FakeClient:
            def undeploy_lora_model(
                self, lora_model_name, *, git_identity, credentials=None
            ):
                captured["lora_model_name"] = lora_model_name
                return _lora_model_summary("qwen3-run1-step-100", status="inactive")

        monkeypatch.setattr(platform_model_module, "OsmosisClient", FakeClient)

        with override_output_context(
            format=OutputFormat.rich, interactive=True
        ) as output:
            monkeypatch.setattr(output, "status", record_spinner)
            result = platform_model_module.undeploy("[red]bad[/red]")

        assert captured["lora_model_name"] == "[red]bad[/red]"
        assert captured["message"] == 'Undeploying LoRA model "\\[red]bad\\[/red]"...'
        assert isinstance(result, OperationResult)
