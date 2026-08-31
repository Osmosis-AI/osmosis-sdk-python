"""`osmosis eval run` CLI-layer tests: config extraction, options, and error UX."""

from __future__ import annotations

import asyncio
import json
import subprocess
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from shlex import quote
from types import SimpleNamespace
from typing import Any

import pytest
import typer
from packaging.requirements import Requirement
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import PipeInput, create_pipe_input
from prompt_toolkit.output import DummyOutput

import osmosis_ai.platform.cli.eval_run as eval_run_module
from osmosis_ai.cli.console import Console
from osmosis_ai.cli.errors import CLIError, CLIErrorCode
from osmosis_ai.cli.output import OutputFormat, override_output_context

GIT_IDENTITY = "acme/rollouts"
ROLLOUT = "echo-rollout"

ROLLOUT_MAIN = """
from osmosis_ai.rollout import AgentWorkflow, Grader


class EchoWorkflow(AgentWorkflow):
    async def run(self, ctx):
        return None


class EchoGrader(Grader):
    async def grade(self, ctx):
        return 1.0
""".strip()

EVAL_CONFIG = """
[experiment]
rollout = "echo-rollout"
entrypoint = "main.py"
model_path = "openai/gpt-5-mini"
dataset = "multiply"

[evaluation]
limit = 3
n = 2
batch_size = 4
pass_threshold = 0.5
agent_workflow_timeout_s = 450
grader_timeout_s = 150

[env]
LOG_LEVEL = "INFO"

[secrets]
required = ["MY_TOKEN"]
""".strip()

DATASET_ROWS = [
    {"user_prompt": "a", "ground_truth": "1"},
    {"user_prompt": "b", "ground_truth": "2"},
    {"user_prompt": "c", "ground_truth": "3"},
    {"user_prompt": "d", "ground_truth": "4"},
]


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    subprocess.run(
        ["git", "init", "-b", "main", str(root)], check=True, capture_output=True
    )
    for rel in (
        ".osmosis",
        f"rollouts/{ROLLOUT}",
        "configs/eval",
        "configs/training",
        "data",
    ):
        (root / rel).mkdir(parents=True, exist_ok=True)
    (root / "rollouts" / ROLLOUT / "main.py").write_text(ROLLOUT_MAIN, encoding="utf-8")
    (root / "configs" / "eval" / "echo.toml").write_text(EVAL_CONFIG, encoding="utf-8")
    (root / "data" / "echo.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in DATASET_ROWS), encoding="utf-8"
    )
    return root


@pytest.fixture
def console_capture(monkeypatch: pytest.MonkeyPatch) -> StringIO:
    output = StringIO()
    monkeypatch.setattr(
        eval_run_module,
        "console",
        Console(file=output, force_terminal=False, width=200),
    )
    return output


@pytest.fixture(autouse=True)
def _stub_context(monkeypatch: pytest.MonkeyPatch, workspace: Path) -> None:
    credentials = type("Credentials", (), {"access_token": "platform-token"})()
    context = type(
        "GitContext",
        (),
        {
            "workspace_directory": workspace.resolve(),
            "git_identity": GIT_IDENTITY,
            "repo_url": f"https://github.com/{GIT_IDENTITY}.git",
            "credentials": credentials,
        },
    )()
    monkeypatch.setattr(
        eval_run_module, "require_git_workspace_directory_context", lambda: context
    )
    monkeypatch.setattr(
        eval_run_module,
        "resolve_local_workspace_directory_context",
        lambda: context,
    )
    monkeypatch.setattr(
        eval_run_module, "validate_workspace_directory_contract", lambda _: None
    )


def _metrics(*, passed: int, scored: int) -> dict[str, Any]:
    """The subset of ``aggregate_metrics`` output the CLI layer reads."""
    return {
        "total_samples": scored,
        "completed_samples": scored,
        "graded": scored,
        "passed": passed,
        "failed": scored - passed,
        "skipped": 0,
        "pass_rate": passed / scored,
        "pass_threshold": 0.5,
        "tokens_used": 4096,
    }


class _CapturedRunner:
    """Stands in for the supervisor and records exactly how it was constructed."""

    calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        type(self).calls.append(kwargs)
        self.kwargs = kwargs

    async def run(self) -> Any:
        from osmosis_ai.eval.local.runner import RunSummary

        return RunSummary(
            run_dir=self.kwargs["output_root"] / "run-1",
            local_run_id="a" * 32,
            run_name="run-1",
            total_work_items=6,
            dispatched=6,
            succeeded=6,
            failed=0,
            skipped=0,
            resumed=0,
            cancelled=False,
            duration_ms=4200.0,
            metrics=_metrics(passed=6, scored=6),
        )


@pytest.fixture
def captured_runner(monkeypatch: pytest.MonkeyPatch) -> type[_CapturedRunner]:
    _CapturedRunner.calls = []
    import osmosis_ai.eval.local.runner as runner_module

    monkeypatch.setattr(runner_module, "LocalEvalRunner", _CapturedRunner)
    return _CapturedRunner


def _run(workspace: Path, **kwargs: Any) -> Any:
    return eval_run_module.run(
        workspace / "configs" / "eval" / "echo.toml",
        dataset_file=str(workspace / "data" / "echo.jsonl"),
        yes=True,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Config extraction
# --------------------------------------------------------------------------- #


def test_the_shared_eval_toml_maps_onto_the_run_spec(
    workspace: Path, captured_runner: type[_CapturedRunner], console_capture: StringIO
) -> None:
    _run(workspace)
    spec = captured_runner.calls[0]["spec"]
    assert spec.rollout_name == ROLLOUT
    assert spec.entrypoint == "main.py"
    assert spec.model_path == "openai/gpt-5-mini"
    assert spec.dataset_name == "multiply"
    assert spec.n == 2
    assert spec.batch_size == 4
    assert spec.pass_threshold == 0.5
    assert spec.agent_timeout_sec == 450.0
    assert spec.grader_timeout_sec == 150.0
    assert spec.env == {"LOG_LEVEL": "INFO"}
    assert spec.secret_names == ("MY_TOKEN",)


def test_dataset_file_run_does_not_require_platform_credentials(
    workspace: Path,
    captured_runner: type[_CapturedRunner],
    console_capture: StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        eval_run_module,
        "require_git_workspace_directory_context",
        lambda: pytest.fail("local dataset run requested platform credentials"),
    )

    result = _run(workspace)

    assert result.status == "success"
    assert len(captured_runner.calls) == 1


def test_evaluation_limit_selects_the_first_rows(
    workspace: Path, captured_runner: type[_CapturedRunner], console_capture: StringIO
) -> None:
    _run(workspace)
    selection = captured_runner.calls[0]["selection"]
    assert selection.source_row_indices == (0, 1, 2)
    assert selection.total_dataset_rows == 4


def test_rows_overrides_the_configured_limit(
    workspace: Path, captured_runner: type[_CapturedRunner], console_capture: StringIO
) -> None:
    _run(workspace, rows="3")
    assert captured_runner.calls[0]["selection"].source_row_indices == (3,)


def test_runtime_flags_reach_the_options_object(
    workspace: Path, captured_runner: type[_CapturedRunner], console_capture: StringIO
) -> None:
    _run(
        workspace,
        name="my-run",
        fresh=True,
        retry_failed=True,
        max_in_flight=7,
        rollout_port=8123,
        verbose=True,
    )
    options = captured_runner.calls[0]["options"]
    assert options.name == "my-run"
    assert options.fresh is True
    assert options.retry_failed is True
    assert options.max_in_flight == 7
    assert options.rollout_port == 8123
    assert options.verbose is True


def test_tunnel_flags_reach_the_options_object(
    workspace: Path, captured_runner: type[_CapturedRunner], console_capture: StringIO
) -> None:
    _run(workspace, tunnel="cloudflared", listener_port=9321)
    options = captured_runner.calls[0]["options"]
    assert options.tunnel == "cloudflared"
    assert options.listener_port == 9321
    assert options.advertise_url is None


def test_advertise_url_reaches_the_options_object(
    workspace: Path, captured_runner: type[_CapturedRunner], console_capture: StringIO
) -> None:
    _run(workspace, advertise_url="https://my-tunnel.example", listener_port=8710)
    options = captured_runner.calls[0]["options"]
    assert options.advertise_url == "https://my-tunnel.example"
    assert options.listener_port == 8710
    assert options.tunnel is None


def test_advertise_url_requires_a_listener_port() -> None:
    # An external tunnel forwards to a fixed local port; an ephemeral bind
    # would leave it nothing stable to target.
    with pytest.raises(CLIError, match="listener-port"):
        eval_run_module.run(
            Path("unused.toml"), advertise_url="https://my-tunnel.example"
        )


def test_unknown_tunnel_provider_fails_before_any_work() -> None:
    with pytest.raises(CLIError, match="only 'cloudflared'"):
        eval_run_module.run(Path("unused.toml"), tunnel="ngrok")


def test_tunnel_and_advertise_url_are_mutually_exclusive() -> None:
    with pytest.raises(CLIError, match="mutually exclusive"):
        eval_run_module.run(
            Path("unused.toml"),
            tunnel="cloudflared",
            advertise_url="https://my-tunnel.example",
        )


def test_advertise_url_must_be_http() -> None:
    with pytest.raises(CLIError, match="http"):
        eval_run_module.run(Path("unused.toml"), advertise_url="my-tunnel.example")


def test_advertise_url_must_carry_a_host() -> None:
    with pytest.raises(CLIError, match="host"):
        eval_run_module.run(Path("unused.toml"), advertise_url="http://")


@pytest.mark.parametrize(
    "url",
    [
        "https://tunnel.example.com?token=x",
        "https://tunnel.example.com/#fragment",
    ],
)
def test_advertise_url_rejects_query_and_fragment(url: str) -> None:
    # The rollout path is appended verbatim, so anything after a query or
    # fragment delimiter would aim the sandbox at the wrong endpoint.
    with pytest.raises(CLIError, match="query or fragment"):
        eval_run_module.run(Path("unused.toml"), advertise_url=url)


def test_follow_up_commands_carry_the_tunnel_flags(
    workspace: Path,
    captured_runner: type[_CapturedRunner],
    console_capture: StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Tunnel flags are runtime-only and never land in the manifest, so a
    # displayed Resume/Retry command without them would point every pending
    # item at the loopback guard.
    from osmosis_ai.eval.local.runner import RunSummary

    async def partial_run(self: Any, **_kwargs: Any) -> RunSummary:
        return RunSummary(
            run_dir=self.kwargs["output_root"] / "run-1",
            local_run_id="a" * 32,
            run_name="run-1",
            total_work_items=6,
            dispatched=3,
            succeeded=2,
            failed=1,
            skipped=0,
            resumed=0,
            cancelled=True,
            duration_ms=1500.0,
            metrics=_metrics(passed=1, scored=2),
        )

    monkeypatch.setattr(captured_runner, "run", partial_run)
    result = _run(workspace, tunnel="cloudflared", listener_port=9321)
    flags = " --tunnel cloudflared --listener-port 9321"
    resume = next(s for s in result.display_next_steps if s.startswith("Resume:"))
    retry = next(s for s in result.display_next_steps if "Retry failures" in s)
    assert resume.endswith(flags)
    assert retry.endswith(flags)


def test_listener_port_must_be_a_real_port() -> None:
    # 99999 would raise OverflowError from socket.bind, far past the friendly
    # error paths.
    with pytest.raises(CLIError, match="between 1 and 65535"):
        eval_run_module.run(Path("unused.toml"), listener_port=99999)


def test_the_default_output_root_is_the_workspace_evals_dir(
    workspace: Path, captured_runner: type[_CapturedRunner], console_capture: StringIO
) -> None:
    _run(workspace)
    assert captured_runner.calls[0]["output_root"] == workspace / ".osmosis" / "evals"


def test_output_override_is_honored(
    workspace: Path,
    captured_runner: type[_CapturedRunner],
    console_capture: StringIO,
    tmp_path: Path,
) -> None:
    _run(workspace, output=str(tmp_path / "elsewhere"))
    assert captured_runner.calls[0]["output_root"] == tmp_path / "elsewhere"


def test_the_rollout_dir_is_resolved_under_the_workspace(
    workspace: Path, captured_runner: type[_CapturedRunner], console_capture: StringIO
) -> None:
    _run(workspace)
    assert captured_runner.calls[0]["rollout_dir"] == workspace / "rollouts" / ROLLOUT


# --------------------------------------------------------------------------- #
# Errors and UX
# --------------------------------------------------------------------------- #


def test_a_malformed_rows_selector_is_a_validation_error(
    workspace: Path, console_capture: StringIO
) -> None:
    with pytest.raises(CLIError, match="--rows"):
        _run(workspace, rows="not-rows")


def test_an_out_of_range_row_fails_before_the_run_exists(
    workspace: Path, console_capture: StringIO
) -> None:
    with pytest.raises(CLIError, match="has 4 rows"):
        _run(workspace, rows="9")


def test_a_missing_dataset_file_is_a_validation_error(
    workspace: Path, console_capture: StringIO
) -> None:
    with pytest.raises(CLIError, match="is not a file"):
        eval_run_module.run(
            workspace / "configs" / "eval" / "echo.toml",
            dataset_file=str(workspace / "data" / "absent.jsonl"),
            yes=True,
        )


def test_the_missing_extra_hint_names_the_install_command() -> None:
    error = eval_run_module._missing_extra_error(ModuleNotFoundError(name="fastapi"))
    assert 'pip install "osmosis-ai[eval]"' in error.message
    assert error.details == {"missing_module": "fastapi", "extra": "eval"}


def test_eval_run_extra_includes_parquet_dataset_support() -> None:
    repo_root = Path(__file__).parents[4]
    pyproject = tomllib.loads(
        (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    )
    requirements = [
        Requirement(value)
        for value in pyproject["project"]["optional-dependencies"]["eval"]
    ]
    sdk_requirement = next(req for req in requirements if req.name == "osmosis-ai")

    assert {"server", "parquet"} <= sdk_requirement.extras


def _with_pass_threshold(workspace: Path, value: str) -> None:
    (workspace / "configs" / "eval" / "echo.toml").write_text(
        EVAL_CONFIG.replace("pass_threshold = 0.5", f"pass_threshold = {value}"),
        encoding="utf-8",
    )


def test_a_zero_pass_threshold_survives_the_default(
    workspace: Path, captured_runner: type[_CapturedRunner], console_capture: StringIO
) -> None:
    # 0.0 means "every graded row passes"; falling back to 1.0 would silently
    # apply the strictest threshold instead.
    _with_pass_threshold(workspace, "0.0")
    _run(workspace)
    assert captured_runner.calls[0]["spec"].pass_threshold == 0.0


@pytest.mark.parametrize("value", ["nan", "inf"])
def test_a_non_finite_pass_threshold_is_a_cli_error(
    workspace: Path, console_capture: StringIO, value: str
) -> None:
    _with_pass_threshold(workspace, value)
    with pytest.raises(CLIError, match=r"evaluation\.pass_threshold must be finite"):
        _run(workspace)


@pytest.mark.parametrize("value", ["0", "-1"])
def test_a_nonpositive_n_is_a_cli_error(
    workspace: Path, console_capture: StringIO, value: str
) -> None:
    config = workspace / "configs" / "eval" / "echo.toml"
    config.write_text(EVAL_CONFIG.replace("n = 2", f"n = {value}"), encoding="utf-8")
    with pytest.raises(CLIError, match=r"evaluation\.n must be a positive integer"):
        _run(workspace)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("n", '"two"', "evaluation.n must be an integer"),
        ("pass_threshold", '"high"', "evaluation.pass_threshold must be a number"),
        ("batch_size", "1.5", "evaluation.batch_size must be an integer"),
    ],
)
def test_non_numeric_evaluation_fields_are_reported(
    workspace: Path, console_capture: StringIO, field: str, value: str, match: str
) -> None:
    config = workspace / "configs" / "eval" / "echo.toml"
    body = EVAL_CONFIG.replace(f"{field} = 2", f"{field} = {value}")
    body = body.replace(f"{field} = 4", f"{field} = {value}")
    body = body.replace(f"{field} = 0.5", f"{field} = {value}")
    config.write_text(body, encoding="utf-8")
    with pytest.raises(CLIError, match=match):
        _run(workspace)


# --------------------------------------------------------------------------- #
# Result envelope
# --------------------------------------------------------------------------- #


def test_a_complete_run_reports_success_with_the_output_path(
    workspace: Path, captured_runner: type[_CapturedRunner], console_capture: StringIO
) -> None:
    result = _run(workspace)
    assert result.operation == "eval.run"
    assert result.status == "success"
    assert result.exit_code == 0
    assert result.resource["run_name"] == "run-1"
    assert result.resource["succeeded"] == 6
    assert result.resource["dataset_source"] == "explicit"
    assert result.resource["output_path"].endswith("run-1")
    assert result.display_next_steps == [
        f"Upload: osmosis eval upload {quote(str(workspace / '.osmosis/evals/run-1'))}"
    ]


def test_upload_flag_uploads_after_finalize_and_surfaces_platform_url(
    workspace: Path,
    captured_runner: type[_CapturedRunner],
    console_capture: StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import osmosis_ai.eval.local.upload as local_upload_module
    import osmosis_ai.platform.cli.eval_upload as eval_upload_module

    events: list[str] = []
    original_close = eval_run_module._ProgressDisplay.close
    original_run = _CapturedRunner.run

    def close_display(self: Any) -> None:
        events.append("close")
        original_close(self)

    async def run_with_callback(self: Any, *, after_finalize: Any) -> Any:
        summary = await original_run(self)
        after_finalize(summary)
        return summary

    def upload_plan(_plan: Any, *, context: Any) -> SimpleNamespace:
        events.append("upload")
        return SimpleNamespace(
            session_id="session-1",
            eval_run_id="eval-1",
            eval_run_name="uploaded-run",
            status="finalized",
            expected_files=2,
            uploaded_files=2,
            platform_url="https://platform.example/evals/eval-1",
        )

    monkeypatch.setattr(eval_run_module._ProgressDisplay, "close", close_display)
    monkeypatch.setattr(_CapturedRunner, "run", run_with_callback)
    monkeypatch.setattr(
        local_upload_module,
        "build_eval_upload_plan",
        lambda run_dir: SimpleNamespace(run_dir=run_dir),
    )
    monkeypatch.setattr(
        eval_upload_module,
        "upload_plan",
        upload_plan,
    )

    result = _run(workspace, upload=True)

    assert result.resource["upload"]["eval_run_id"] == "eval-1"
    assert result.resource["platform_url"] == "https://platform.example/evals/eval-1"
    assert result.display_next_steps == ["View: https://platform.example/evals/eval-1"]
    assert events[:2] == ["close", "upload"]


def test_upload_failure_says_local_results_are_complete_and_gives_retry_command(
    workspace: Path,
    captured_runner: type[_CapturedRunner],
    console_capture: StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import osmosis_ai.eval.local.upload as local_upload_module
    import osmosis_ai.platform.cli.eval_upload as eval_upload_module

    original_run = _CapturedRunner.run

    async def run_with_callback(self: Any, *, after_finalize: Any) -> Any:
        summary = await original_run(self)
        after_finalize(summary)
        return summary

    monkeypatch.setattr(_CapturedRunner, "run", run_with_callback)
    monkeypatch.setattr(
        local_upload_module,
        "build_eval_upload_plan",
        lambda run_dir: SimpleNamespace(run_dir=run_dir),
    )
    monkeypatch.setattr(
        eval_upload_module,
        "upload_plan",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    run_dir = workspace / ".osmosis" / "evals" / "run-1"

    with pytest.raises(CLIError) as raised:
        _run(workspace, upload=True)

    assert f"Local evaluation results are complete at {run_dir}" in raised.value.message
    assert "local evaluation upload failed: offline" in raised.value.message
    assert f"osmosis eval upload {quote(str(run_dir))}" in raised.value.message
    assert "platform upload failed" not in raised.value.message


def test_upload_plan_error_reports_local_problem_without_retry_guidance(
    workspace: Path,
    captured_runner: type[_CapturedRunner],
    console_capture: StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import osmosis_ai.eval.local.upload as local_upload_module

    original_run = _CapturedRunner.run

    async def run_with_callback(self: Any, *, after_finalize: Any) -> Any:
        summary = await original_run(self)
        after_finalize(summary)
        return summary

    monkeypatch.setattr(_CapturedRunner, "run", run_with_callback)
    monkeypatch.setattr(
        local_upload_module,
        "build_eval_upload_plan",
        lambda _run_dir: (_ for _ in ()).throw(
            local_upload_module.LocalEvalUploadError("index.jsonl is invalid")
        ),
    )

    with pytest.raises(CLIError) as raised:
        _run(workspace, upload=True)

    assert "cannot be uploaded: index.jsonl is invalid" in raised.value.message
    assert "Retry with:" not in raised.value.message
    assert "platform upload failed" not in raised.value.message


def test_unexpected_upload_error_keeps_internal_classification(
    workspace: Path,
    captured_runner: type[_CapturedRunner],
    console_capture: StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import osmosis_ai.eval.local.upload as local_upload_module
    import osmosis_ai.platform.cli.eval_upload as eval_upload_module

    class UnexpectedUploadError(Exception):
        pass

    original_run = _CapturedRunner.run

    async def run_with_callback(self: Any, *, after_finalize: Any) -> Any:
        summary = await original_run(self)
        after_finalize(summary)
        return summary

    monkeypatch.setattr(_CapturedRunner, "run", run_with_callback)
    monkeypatch.setattr(
        local_upload_module,
        "build_eval_upload_plan",
        lambda run_dir: SimpleNamespace(run_dir=run_dir),
    )
    monkeypatch.setattr(
        eval_upload_module,
        "upload_plan",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            UnexpectedUploadError("unexpected")
        ),
    )

    with pytest.raises(CLIError) as raised:
        _run(workspace, upload=True)

    assert raised.value.code == CLIErrorCode.INTERNAL
    assert raised.value.details == {"exception_type": "UnexpectedUploadError"}
    assert "the upload failed: unexpected" in raised.value.message


def test_upload_platform_error_keeps_auth_code_and_retry_context(
    workspace: Path,
    captured_runner: type[_CapturedRunner],
    console_capture: StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import osmosis_ai.eval.local.upload as local_upload_module
    import osmosis_ai.platform.cli.eval_upload as eval_upload_module
    from osmosis_ai.platform.auth.platform_client import PlatformAPIError

    original_run = _CapturedRunner.run

    async def run_with_callback(self: Any, *, after_finalize: Any) -> Any:
        summary = await original_run(self)
        after_finalize(summary)
        return summary

    monkeypatch.setattr(_CapturedRunner, "run", run_with_callback)
    monkeypatch.setattr(
        local_upload_module,
        "build_eval_upload_plan",
        lambda run_dir: SimpleNamespace(run_dir=run_dir),
    )
    monkeypatch.setattr(
        eval_upload_module,
        "upload_plan",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            PlatformAPIError("session expired", status_code=401)
        ),
    )
    run_dir = workspace / ".osmosis" / "evals" / "run-1"

    with pytest.raises(CLIError) as raised:
        _run(workspace, upload=True)

    assert raised.value.code == CLIErrorCode.AUTH_REQUIRED
    assert raised.value.details["status_code"] == 401
    assert f"Local evaluation results are complete at {run_dir}" in raised.value.message
    assert f"osmosis eval upload {quote(str(run_dir))}" in raised.value.message


def test_an_incomplete_upload_run_reports_skipped_upload_and_resumes_with_upload(
    workspace: Path,
    captured_runner: type[_CapturedRunner],
    console_capture: StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osmosis_ai.eval.local.runner import FailedWorkItem, RunSummary

    async def partial_run(self: Any, *, after_finalize: Any) -> RunSummary:
        return RunSummary(
            run_dir=self.kwargs["output_root"] / "run-1",
            local_run_id="a" * 32,
            run_name="run-1",
            total_work_items=6,
            dispatched=3,
            succeeded=2,
            failed=1,
            skipped=0,
            resumed=0,
            cancelled=True,
            duration_ms=1500.0,
            metrics=_metrics(passed=1, scored=2),
            failures=[
                FailedWorkItem(
                    row_index=1,
                    source_row_index=4,
                    run_index=0,
                    rollout_id="b" * 32,
                    error_type="grader_timeout",
                    rollout_dir=Path("/runs/rollout_trials/bbbb"),
                )
            ],
        )

    monkeypatch.setattr(captured_runner, "run", partial_run)
    result = _run(workspace, upload=True)
    assert result.status == "partial"
    assert result.exit_code == 1
    assert "Upload skipped" in result.message
    assert "upload" not in result.resource
    assert any("--name run-1 --upload" in step for step in result.display_next_steps)
    assert any("--retry-failed --upload" in step for step in result.display_next_steps)
    assert result.resource["failed_rows"][0]["error_type"] == "grader_timeout"
    printed = console_capture.getvalue()
    assert "row 1 (source 4) run 0" in printed
    assert "error_type=grader_timeout" in printed
    assert "/runs/rollout_trials/bbbb" in printed


# --------------------------------------------------------------------------- #
# What a plain (non---verbose) run prints
# --------------------------------------------------------------------------- #


def test_the_plan_is_printed_before_the_run_starts(
    workspace: Path, captured_runner: type[_CapturedRunner], console_capture: StringIO
) -> None:
    """The same reading of the TOML `eval submit` gives, so the confirmation is
    answered against what will actually run."""
    _run(workspace)
    printed = console_capture.getvalue()
    assert "Local Evaluation" in printed
    assert "openai/gpt-5-mini" in printed
    assert "echo.jsonl (explicit)" in printed
    assert "3 of 4" in printed  # rows selected of the dataset
    work_items = next(line for line in printed.splitlines() if "Work Items" in line)
    assert "6" in work_items  # 3 rows x n=2


def test_the_results_table_reports_the_metrics(
    workspace: Path, captured_runner: type[_CapturedRunner], console_capture: StringIO
) -> None:
    _run(workspace)
    printed = console_capture.getvalue()
    assert "Results" in printed
    assert "Pass Rate" in printed
    assert "100.0%" in printed
    assert "6/6 complete" in printed
    # The same formatter `eval info` uses, so local and cloud runs read alike.
    assert "Duration" in printed
    assert "4.2s" in printed


def test_stage_lines_print_once_and_only_without_verbose(
    console_capture: StringIO,
) -> None:
    """--verbose echoes every log line, and each stage is one of them."""
    with override_output_context(format=OutputFormat.rich, interactive=False):
        eval_run_module._Hooks(yes=True, secrets_file=None).stage("preflight ok")
        eval_run_module._Hooks(yes=True, secrets_file=None).stage(
            "checking model [/bold]"
        )
        eval_run_module._Hooks(yes=True, secrets_file=None, verbose=True).stage(
            "not repeated"
        )
    printed = console_capture.getvalue()
    assert "preflight ok" in printed
    assert "checking model [/bold]" in printed
    assert "not repeated" not in printed


def test_runner_warning_uses_the_output_mode_aware_console(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        eval_run_module.console,
        "print_warning",
        lambda message, *, code=None: calls.append((message, code)),
    )

    eval_run_module._Hooks(yes=True, secrets_file=None).warning("versions differ")
    with override_output_context(format=OutputFormat.rich):
        eval_run_module._Hooks(yes=True, secrets_file=None, verbose=True).warning(
            "already streamed"
        )

    assert calls == [("versions differ", "ROLLOUT_SDK_VERSION_MISMATCH")]


def test_startup_status_uses_spinner_except_in_verbose_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    @contextmanager
    def fake_status(message: str) -> Iterator[None]:
        events.append(f"start:{message}")
        try:
            yield
        finally:
            events.append(f"stop:{message}")

    monkeypatch.setattr(eval_run_module.console, "status", fake_status)
    with eval_run_module._Hooks(yes=True, secrets_file=None).status("starting"):
        events.append("work")
    with eval_run_module._Hooks(yes=True, secrets_file=None, verbose=True).status(
        "verbose"
    ):
        events.append("verbose-work")

    assert events == ["start:starting", "work", "stop:starting", "verbose-work"]


def test_progress_falls_back_to_printed_lines_without_a_terminal(
    console_capture: StringIO,
) -> None:
    """Redirected output gets one line per completion, not a live region."""
    from osmosis_ai.eval.local.runner import ProgressSnapshot

    display = eval_run_module._ProgressDisplay()
    with override_output_context(format=OutputFormat.rich, interactive=False):
        display.update(ProgressSnapshot(completed=2, total=6, passed=1, failed=1))
        display.close()
    assert "2/6 pass 50% · failed 1" in console_capture.getvalue()


# --------------------------------------------------------------------------- #
# Through the CLI entry point
# --------------------------------------------------------------------------- #


def test_the_json_envelope_carries_the_run_resource(
    workspace: Path,
    captured_runner: type[_CapturedRunner],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import osmosis_ai.cli.main as cli

    monkeypatch.chdir(workspace)
    exit_code = cli.main(
        [
            "--json",
            "eval",
            "run",
            "configs/eval/echo.toml",
            "--dataset-file",
            "data/echo.jsonl",
            "--name",
            "run-1",
            "--yes",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0, captured.err
    envelope = json.loads(captured.out)
    assert envelope["status"] == "success"
    assert envelope["operation"] == "eval.run"
    assert envelope["resource"]["run_name"] == "run-1"
    assert envelope["resource"]["metrics"]["pass_rate"] == 1.0
    assert envelope["resource"]["duration_ms"] == 4200.0


def test_plain_output_prints_the_upload_command_after_a_complete_run(
    workspace: Path,
    captured_runner: type[_CapturedRunner],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import osmosis_ai.cli.main as cli

    monkeypatch.chdir(workspace)
    exit_code = cli.main(
        [
            "--plain",
            "eval",
            "run",
            "configs/eval/echo.toml",
            "--dataset-file",
            "data/echo.jsonl",
            "--name",
            "run-1",
            "--yes",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0, captured.err
    assert (
        f"Upload: osmosis eval upload {quote(str(workspace / '.osmosis/evals/run-1'))}"
        in captured.out
    )


def test_the_command_is_registered_under_the_eval_group() -> None:
    from osmosis_ai.cli.commands.eval import app

    names = {
        command.name or command.callback.__name__  # type: ignore[union-attr]
        for command in app.registered_commands
    }
    assert "run" in names
    assert "upload" in names


# ── The dispatch confirmation, on the supervisor's own loop ──────────

# A prompt that never answers hangs the suite, so bound every ask.
ASK_TIMEOUT = 10.0


@pytest.fixture
def keys() -> Iterator[PipeInput]:
    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=DummyOutput()):
            yield pipe_input


async def _confirm_dispatch() -> None:
    hooks = eval_run_module._Hooks(yes=False, secrets_file=None)
    with override_output_context(format=OutputFormat.rich, interactive=True):
        await asyncio.wait_for(
            hooks.confirm_dispatch(pending=4, model_path="openai/gpt-5-mini"),
            timeout=ASK_TIMEOUT,
        )


async def test_the_dispatch_confirmation_prompts_on_the_running_loop(
    keys: PipeInput,
) -> None:
    """The supervisor awaits this hook from inside ``asyncio.run``; a prompt
    that ran on a loop of its own would raise ``RuntimeError`` there."""
    keys.send_text("y")
    await _confirm_dispatch()


async def test_declining_the_dispatch_confirmation_exits_before_any_rollout(
    keys: PipeInput,
) -> None:
    keys.send_text("n")
    with pytest.raises(typer.Exit) as raised:
        await _confirm_dispatch()
    assert raised.value.exit_code == 0
