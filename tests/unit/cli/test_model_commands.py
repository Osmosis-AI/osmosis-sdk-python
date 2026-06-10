"""Tests for osmosis_ai.cli.commands.model."""

from __future__ import annotations

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
    ListResult,
    OperationResult,
    OutputFormat,
    override_output_context,
)
from osmosis_ai.platform.api.models import (
    BaseModelInfo,
    LoraModelInfo,
    LoraModelSummary,
    PaginatedBaseModels,
    PaginatedLoraModels,
)

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
    assert "Osmosis workspace directory" in capsys.readouterr().err


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
            )

    return FakeClient


@pytest.mark.usefixtures("mock_git_context")
class TestListModels:
    def test_empty_list(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        monkeypatch.setattr(
            platform_model_module, "OsmosisClient", _fake_list_client([], [])
        )
        result = platform_model_module.list_models(limit=30, all_=False)

        assert isinstance(result, ListResult)
        assert result.title == "Models"
        assert result.items == []
        assert result.total_count == 0
        assert result.has_more is False
        assert result.next_offset is None
        assert result.display_hints == []
        assert_git_context(result.extra)

    def test_list_combines_base_models_before_lora_models(
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
        assert isinstance(result, ListResult)
        assert [(i["type"], i["model_name"]) for i in result.items] == [
            ("base", "Qwen/Qwen3"),
            ("lora", "qwen3-run1-step-100"),
        ]
        assert "status" not in result.items[0]
        assert result.items[1]["deployment_status"] == "active"
        assert result.items[1]["checkpoint_step"] == 100
        assert result.items[1]["training_run_name"] == "qwen3-run1"
        assert result.total_count == 2
        assert result.display_hints == [
            "Deploy a LoRA model with: osmosis model deploy <name>"
        ]

    def test_list_display_items_fill_missing_fields_with_dashes(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        monkeypatch.setattr(
            platform_model_module,
            "OsmosisClient",
            _fake_list_client(
                [_base_model()],
                [
                    _lora_model(
                        base_model=None,
                        training_run_name=None,
                        checkpoint_step=None,
                        deployment_status=None,
                    )
                ],
            ),
        )
        result = platform_model_module.list_models(limit=30, all_=False)

        assert result.display_items is not None
        base_display, lora_display = result.display_items
        assert base_display["type"] == "Base"
        assert base_display["deployment_status"] == "—"
        assert base_display["training_run_name"] == "—"
        assert base_display["checkpoint_step"] == "—"
        assert lora_display["type"] == "LoRA"
        assert lora_display["deployment_status"] == "—"
        assert lora_display["base_model"] == "—"
        assert lora_display["checkpoint_step"] == "—"
        # Raw JSON items keep nulls — display dashes must not leak into them.
        assert result.items[1]["deployment_status"] is None
        assert result.items[1]["checkpoint_step"] is None

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
        assert [i["type"] for i in result.items] == ["base"]
        assert result.total_count == 1
        assert result.display_hints == []

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
        assert [i["type"] for i in result.items] == ["lora"]
        assert result.next_offset == 1

    def test_list_type_all_has_no_continuation_cursor(
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
                lora_next_offset=1,
            ),
        )
        result = platform_model_module.list_models(limit=30, all_=False)

        assert result.has_more is True
        assert result.next_offset is None

    def test_list_rejects_invalid_type(self) -> None:
        with pytest.raises(CLIError, match=r"Type must be 'all', 'base', or 'lora'\."):
            platform_model_module.list_models(limit=30, all_=False, type_="bogus")


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
        assert result.message == "LoRA model qwen3-run1-step-100 active"

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

    def test_deploy_failed_result_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeClient:
            def deploy_lora_model(
                self, lora_model_name, *, git_identity, credentials=None
            ):
                assert git_identity == GIT_IDENTITY
                return _lora_model_summary("qwen3-run1-step-100", status="failed")

        monkeypatch.setattr(platform_model_module, "OsmosisClient", FakeClient)
        result = platform_model_module.deploy("qwen3-run1-step-100")

        assert isinstance(result, OperationResult)
        assert result.status == "failed"
        assert result.message == "LoRA model qwen3-run1-step-100 failed"
        assert result.exit_code == 1


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
        assert result.message == "LoRA model qwen3-run1-step-100 inactive"

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
