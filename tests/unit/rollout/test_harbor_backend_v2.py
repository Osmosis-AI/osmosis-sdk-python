"""Bundle backend: contract round-trip, task materialization, trial config."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from harbor.trial.queue import TrialQueue

from osmosis_ai.packaging import build_bundle, inspect_bundle
from osmosis_ai.rollout.backend.harbor.backend import PendingTrial
from osmosis_ai.rollout.backend.harbor.backend_v2 import HarborBackendV2
from osmosis_ai.rollout.backend.harbor.tasks import (
    SDK_REQUIREMENTS_FILENAME,
    HarborTask,
    TaskMode,
    patch_dockerfile_with_sdk,
    venv_or_fallback_script,
)
from osmosis_ai.rollout.container.files import ContainerInput, ContainerResult
from osmosis_ai.rollout.types import (
    ExecutionRequest,
    RolloutErrorCategory,
    RolloutStatus,
)
from osmosis_ai.rollout.types.output import AgentWorkflowOutput, coerce_output

PYPROJECT = """\
[project]
name = "bench"
version = "0.1.0"
dependencies = []

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["bench*"]
"""


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    code_dir = tmp_path_factory.mktemp("harness") / "project"
    package = code_dir / "bench"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    (package / "solver.py").write_text("class W: pass\nclass G: pass\n")
    (code_dir / "pyproject.toml").write_text(PYPROJECT)
    return build_bundle(
        code_dir,
        workflow="bench.solver:W",
        grader="bench.solver:G",
        bundles_dir=tmp_path_factory.mktemp("bundles"),
    )


@pytest.fixture
def template_task(tmp_path):
    task = tmp_path / "template-task"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (task / "task.toml").write_text('[task]\nname = "template-task"\n')
    return task


def request_for(prompt=None, metadata=None) -> ExecutionRequest:
    return ExecutionRequest(id="r1", prompt=prompt or [], metadata=metadata)


class TestContract:
    def test_spec_round_trip(self, tmp_path):
        container_input = ContainerInput(
            rollout_id="r1",
            prompt=[{"role": "user", "content": "hi"}],
            label="4",
            metadata={"k": 1},
            chat_completions_url="http://x/v1",
            api_key="secret",
        )
        container_input.write(tmp_path / "input.json")
        assert ContainerInput.read(tmp_path / "input.json") == container_input

    def test_result_round_trip(self, tmp_path):
        output = AgentWorkflowOutput(
            samples={"default": [{"role": "assistant", "content": "y"}]},
            metrics={"turns": 1.0},
        )
        result = ContainerResult(status=RolloutStatus.SUCCESS, output=output)
        result.write(tmp_path / "result.json")
        loaded = ContainerResult.read(tmp_path / "result.json")
        assert loaded.status == RolloutStatus.SUCCESS
        assert loaded.output.primary_messages() == output.primary_messages()
        assert loaded.output.metrics == {"turns": 1.0}


class TestAgentWorkflowOutput:
    def test_coerce_none_passes_through(self):
        assert coerce_output(None) is None

    def test_coerce_messages_wraps_as_default(self):
        messages = [{"role": "assistant", "content": "hi"}]
        output = coerce_output(messages)
        assert output.samples == {"default": messages}
        assert output.primary_messages() == messages

    def test_coerce_output_object_passes_through(self):
        output = AgentWorkflowOutput(samples={"solver": [], "critic": []})
        assert coerce_output(output) is output

    def test_coerce_rejects_other_types(self):
        import pytest as pytest_module

        with pytest_module.raises(TypeError, match="run\\(\\) must return"):
            coerce_output("a string")

    def test_primary_prefers_default_key(self):
        default = [{"role": "assistant", "content": "d"}]
        output = AgentWorkflowOutput(samples={"z": [], "default": default})
        assert output.primary_messages() == default

    def test_empty_output_has_no_primary(self):
        assert AgentWorkflowOutput().primary_messages() is None


class TestHarborTask:
    def test_template_writes_prompt_and_input(self, template_task, tmp_path):
        prompt = [{"role": "user", "content": "solve"}]
        container_input = ContainerInput(rollout_id="r1", prompt=prompt)

        task_dir = HarborTask(template_task).materialize(
            tmp_path / "r1", container_input
        )

        assert json.loads((task_dir / "instruction.md").read_text()) == prompt
        assert ContainerInput.read(task_dir / "container_input.json") == container_input
        assert (task_dir / "environment" / "Dockerfile").exists()

    def test_without_prompt_or_instruction_rejected(self, template_task, tmp_path):
        with pytest.raises(ValueError, match="no instruction"):
            HarborTask(template_task).materialize(
                tmp_path / "r1", ContainerInput(rollout_id="r1")
            )

    def test_grader_script_generates_test_sh(self, template_task, tmp_path):
        prompt = [{"role": "user", "content": "x"}]
        task_dir = HarborTask(template_task).materialize(
            tmp_path / "r1",
            ContainerInput(rollout_id="r1", prompt=prompt, label="42"),
            grader_script="bench-grade",
        )
        assert "bench-grade" in (task_dir / "tests" / "test.sh").read_text()
        staged = ContainerInput.read(task_dir / "tests" / "container_input.json")
        assert staged.label == "42"

    def test_task_native_tests_win(self, template_task, tmp_path):
        (template_task / "tests").mkdir()
        (template_task / "tests" / "test.sh").write_text("#!/bin/bash\nnative\n")
        prompt = [{"role": "user", "content": "x"}]
        task_dir = HarborTask(template_task).materialize(
            tmp_path / "r1",
            ContainerInput(rollout_id="r1", prompt=prompt),
            grader_script="bench-grade",
        )
        assert "native" in (task_dir / "tests" / "test.sh").read_text()

    def test_from_dataset_routes_by_task_id(self, tmp_path):
        root = tmp_path / "dataset"
        task = root / "task-a"
        (task / "environment").mkdir(parents=True)
        (task / "instruction.md").write_text("fix the bug")

        task_dir = HarborTask.from_dataset(root, "task-a").materialize(
            tmp_path / "r1", ContainerInput(rollout_id="r1")
        )
        assert (task_dir / "instruction.md").read_text() == "fix the bug"

    def test_from_dataset_rejects_traversal_and_unknown_ids(self, tmp_path):
        root = tmp_path / "dataset"
        root.mkdir()
        with pytest.raises(ValueError, match="unknown harbor task id"):
            HarborTask.from_dataset(root, "../../etc")
        with pytest.raises(ValueError, match="unknown harbor task id"):
            HarborTask.from_dataset(root, "nope")


class TestHarnessAgentLabelStrip:
    async def test_agent_phase_copy_has_no_label(self, tmp_path):
        from osmosis_ai.rollout.backend.harbor.harness_agent import (
            OsmosisHarnessInstalledAgent,
        )

        agent = OsmosisHarnessInstalledAgent.__new__(OsmosisHarnessInstalledAgent)
        agent.logs_dir = tmp_path / "logs"
        agent.logs_dir.mkdir()
        agent.input_path = tmp_path / "container_input.json"
        agent.agent_script = "bench-agent"
        ContainerInput(rollout_id="r1", label="42").write(agent.input_path)

        async def fake_exec(environment, command):
            pass

        agent.exec_as_agent = fake_exec
        env = SimpleNamespace(capabilities=SimpleNamespace(mounted=True))

        await agent.run("do it", env, None)

        staged = ContainerInput.read(agent.logs_dir / "container_input.json")
        assert staged.label is None
        assert staged.prompt == [{"role": "user", "content": "do it"}]
        assert ContainerInput.read(agent.input_path).label == "42"


class TestPatchDockerfileWithSdk:
    def test_patch_appends_isolated_venv(self, tmp_path):
        env = tmp_path / "environment"
        env.mkdir()
        (env / "Dockerfile").write_text(
            "FROM builder AS build\nRUN make\n"
            'FROM python:3.12-slim\nUSER agent\nCMD ["bash"]\n'
        )
        patch_dockerfile_with_sdk(env, ["pydantic>=2", "httpx"])

        dockerfile = (env / "Dockerfile").read_text()
        reqs = (env / SDK_REQUIREMENTS_FILENAME).read_text()
        assert reqs == "pydantic>=2\nhttpx\n"
        assert "uv venv /opt/osmosis/venv" in dockerfile
        # USER root is scoped to the install; the stage's user is restored last
        assert dockerfile.rstrip().endswith("USER agent")
        assert dockerfile.index("USER root") < dockerfile.index("uv venv")

    def test_patch_without_final_user_adds_no_restore(self, tmp_path):
        env = tmp_path / "environment"
        env.mkdir()
        (env / "Dockerfile").write_text("FROM python:3.12-slim\n")
        patch_dockerfile_with_sdk(env, ["httpx"])
        assert (env / "Dockerfile").read_text().count("USER") == 1

    def test_patch_negates_dockerignore(self, tmp_path):
        env = tmp_path / "environment"
        env.mkdir()
        (env / "Dockerfile").write_text("FROM python:3.12-slim\n")
        (env / ".dockerignore").write_text("*\n")
        patch_dockerfile_with_sdk(env, ["httpx"])
        assert f"!{SDK_REQUIREMENTS_FILENAME}" in (env / ".dockerignore").read_text()

    def test_patch_requires_dockerfile(self, tmp_path):
        with pytest.raises(ValueError, match="cannot patch Dockerfile"):
            patch_dockerfile_with_sdk(tmp_path, ["httpx"])

    def test_materialize_patches_dockerfile(self, template_task, tmp_path):
        task_dir = HarborTask(template_task).materialize(
            tmp_path / "r1",
            ContainerInput(rollout_id="r1", prompt=[{"role": "user", "content": "x"}]),
            sdk_requirements=["httpx"],
        )
        assert "uv venv" in (task_dir / "environment" / "Dockerfile").read_text()
        # the source task stays pristine
        assert (
            "uv venv" not in (template_task / "environment" / "Dockerfile").read_text()
        )

    def test_bundle_requirements_skips_extras(self, bundle):
        assert inspect_bundle(bundle).requirements == []

    def test_backend_flag_requires_bundle(self, template_task):
        with pytest.raises(ValueError, match="requires a bundle"):
            HarborBackendV2(
                orchestrator=TrialQueue(n_concurrent=1),
                tasks_dir=template_task,
                agent="mini-swe-agent",
                patch_dockerfile_with_sdk=True,
            )

    def test_patch_defaults_on_with_bundle(self, bundle, template_task):
        backend = HarborBackendV2(
            orchestrator=TrialQueue(n_concurrent=1),
            tasks_dir=template_task,
            bundle=bundle,
        )
        assert backend.sdk_requirements is not None

    def test_patch_defaults_off_without_bundle(self, template_task):
        backend = HarborBackendV2(
            orchestrator=TrialQueue(n_concurrent=1),
            tasks_dir=template_task,
            agent="mini-swe-agent",
        )
        assert backend.sdk_requirements is None

    def test_queue_capacity_bound(self, bundle, template_task):
        backend = HarborBackendV2(
            orchestrator=TrialQueue(n_concurrent=1),
            tasks_dir=template_task,
            bundle=bundle,
            max_queue_depth=2,
        )
        assert backend.has_capacity()
        backend.pending = {"a": None, "b": None, "c": None}
        backend.running = 2
        assert backend.has_capacity()
        backend.running = 1
        assert not backend.has_capacity()
        assert backend.health()["max_queue_depth"] == 2

    def test_unbounded_queue_always_has_capacity(self, bundle, template_task):
        backend = HarborBackendV2(
            orchestrator=TrialQueue(n_concurrent=1),
            tasks_dir=template_task,
            bundle=bundle,
        )
        backend.pending = {str(i): None for i in range(100)}
        assert backend.has_capacity()

    def test_patch_opt_out(self, bundle, template_task):
        backend = HarborBackendV2(
            orchestrator=TrialQueue(n_concurrent=1),
            tasks_dir=template_task,
            bundle=bundle,
            patch_dockerfile_with_sdk=False,
        )
        assert backend.sdk_requirements is None


class TestBundleBackend:
    @pytest.fixture
    def backend(self, bundle, template_task):
        return HarborBackendV2(
            orchestrator=TrialQueue(n_concurrent=1),
            tasks_dir=template_task,
            bundle=bundle,
        )

    def prepare(self, backend, request):
        container_input = backend.build_input(request)
        task_dir = backend.select_task(request).materialize(
            backend.rollouts_dir / request.id,
            container_input,
            grader_script=backend.bundle.grader_script if backend.bundle else None,
        )
        return task_dir, container_input

    def test_trial_config_wires_harness_agent(self, backend):
        request = request_for([{"role": "user", "content": "x"}])
        task_dir, container_input = self.prepare(backend, request)
        config = backend.build_trial_config(task_dir, request, container_input)

        assert config.agent.import_path.endswith(":OsmosisHarnessInstalledAgent")
        assert config.agent.kwargs["agent_script"] == "bench-agent"
        assert config.agent.kwargs["bundle_path"] == str(backend.bundle.wheel)
        assert config.verifier.disable is False  # generated test.sh enables it

    def test_verifier_disabled_without_tests(self, bundle, template_task):
        backend = HarborBackendV2(
            orchestrator=TrialQueue(n_concurrent=1),
            tasks_dir=template_task,
            bundle=bundle,
        )
        request = request_for([{"role": "user", "content": "x"}])
        container_input = backend.build_input(request)
        task_dir = backend.select_task(request).materialize(
            backend.rollouts_dir / request.id, container_input
        )
        config = backend.build_trial_config(task_dir, request, container_input)
        assert config.verifier.disable is True

    def test_build_input_carries_request_fields(self, backend):
        request = ExecutionRequest(
            id="r9",
            prompt=[{"role": "user", "content": "q"}],
            label="42",
            metadata={"harbor_task_id": "t"},
        )
        container_input = backend.build_input(request)
        assert container_input.rollout_id == "r9"
        assert container_input.label == "42"
        assert container_input.metadata == {"harbor_task_id": "t"}

    def test_environment_config_cloned_per_trial(self, backend):
        request = request_for([{"role": "user", "content": "x"}])
        task_dir, container_input = self.prepare(backend, request)
        first = backend.build_trial_config(task_dir, request, container_input)
        second = backend.build_trial_config(task_dir, request, container_input)
        assert first.environment is not second.environment
        assert first.environment is not backend.environment_config


class TestNativeAgents:
    def backend_for(self, agent, template_task, **kwargs):
        return HarborBackendV2(
            orchestrator=TrialQueue(n_concurrent=1),
            tasks_dir=template_task,
            agent=agent,
            **kwargs,
        )

    def test_unknown_native_agent_rejected(self, template_task):
        with pytest.raises(ValueError, match="unknown native agent"):
            self.backend_for("claude-code", template_task)

    def test_env_wired_agent_receives_endpoint(self, template_task):
        backend = self.backend_for("mini-swe-agent", template_task)
        container_input = ContainerInput(
            rollout_id="r1",
            chat_completions_url="http://trainer:30000/sessions/abc/v1",
            api_key="k",
        )
        config = backend.build_agent_config(template_task, container_input)

        assert config.name == "mini-swe-agent"
        assert config.env["OPENAI_API_BASE"] == "http://trainer:30000/sessions/abc/v1"
        assert config.env["OPENAI_API_KEY"] == "k"
        assert config.env["MSWEA_COST_TRACKING"] == "ignore_errors"

    def test_kwargs_wired_agent_receives_endpoint(self, template_task):
        backend = self.backend_for("terminus-2", template_task, model_name="openai/m")
        container_input = ContainerInput(
            rollout_id="r1", chat_completions_url="http://t/v1", api_key="rk-1"
        )
        config = backend.build_agent_config(template_task, container_input)

        assert config.name == "terminus-2"
        assert config.model_name == "openai/m"
        assert config.kwargs["api_base"] == "http://t/v1"
        assert config.kwargs["enable_summarize"] is False
        # Terminus-2 ignores a top-level api_key; it must ride in llm_kwargs.
        assert "api_key" not in config.kwargs
        assert config.kwargs["llm_kwargs"]["api_key"] == "rk-1"
        assert config.kwargs["llm_kwargs"]["extra_body"]["stream"] is False

    def test_kwargs_wiring_merges_user_llm_kwargs_without_mutation(self):
        from osmosis_ai.rollout.backend.harbor.native_agents import (
            NativeAgentBinding,
            native_agent_config,
        )

        binding = NativeAgentBinding(
            wiring="kwargs",
            kwargs={
                "enable_summarize": False,
                "llm_kwargs": {
                    "api_key": "user-key",
                    "temperature": 0.2,
                    "extra_body": {"stream": True, "top_k": 5},
                },
            },
        )
        config = native_agent_config(
            "terminus-2", binding, "openai/m", "http://t/v1", "rk-1"
        )

        llm_kwargs = config.kwargs["llm_kwargs"]
        assert llm_kwargs["api_key"] == "rk-1"
        assert llm_kwargs["temperature"] == 0.2
        assert llm_kwargs["extra_body"] == {"stream": False, "top_k": 5}
        # The registered binding must stay pristine for the next rollout.
        assert binding.kwargs["llm_kwargs"]["api_key"] == "user-key"
        assert binding.kwargs["llm_kwargs"]["extra_body"]["stream"] is True

    def test_native_without_grader_needs_no_bundle(self, template_task):
        backend = self.backend_for("mini-swe-agent", template_task)
        assert backend.bundle is None

    def test_dataset_mode_keeps_task_instruction(self, tmp_path, bundle):
        root = tmp_path / "dataset"
        task = root / "task-a"
        (task / "environment").mkdir(parents=True)
        (task / "instruction.md").write_text("real instruction")
        backend = HarborBackendV2(
            orchestrator=TrialQueue(n_concurrent=1),
            tasks_dir=root,
            task_mode=TaskMode.DATASET,
            bundle=bundle,
        )
        request = request_for(
            [{"role": "user", "content": "row prompt"}],
            metadata={"harbor_task_id": "task-a"},
        )
        container_input = backend.build_input(request)
        assert container_input.prompt == []

        task_dir = backend.select_task(request).materialize(
            backend.rollouts_dir / request.id, container_input
        )
        assert (task_dir / "instruction.md").read_text() == "real instruction"

    def test_grader_wheel_ships_in_tests_dir(self, template_task, tmp_path, bundle):
        from osmosis_ai.packaging import inspect_bundle

        info = inspect_bundle(bundle)
        prompt = [{"role": "user", "content": "x"}]
        task_dir = HarborTask(template_task).materialize(
            tmp_path / "r1",
            ContainerInput(rollout_id="r1", prompt=prompt),
            grader_script=info.grader_script,
            grader_wheel=info.wheel,
        )
        test_sh = (task_dir / "tests" / "test.sh").read_text()
        assert f"pip install /tests/{info.wheel.name}" in test_sh
        # the grader script runs from the SDK venv when the image has one
        assert test_sh.rstrip().endswith(venv_or_fallback_script(info.grader_script))
        assert (task_dir / "tests" / info.wheel.name).exists()
        assert (task_dir / "tests" / "container_input.json").exists()

    def test_oracle_binding_gets_no_endpoint(self, template_task):
        backend = self.backend_for("oracle", template_task)
        config = backend.build_agent_config(
            template_task,
            ContainerInput(rollout_id="r1", chat_completions_url="http://t/v1"),
        )
        assert config.name == "oracle"
        assert "OPENAI_API_BASE" not in config.env
        assert "api_base" not in config.kwargs

    def test_harbor_model_metadata_overrides_model(self, template_task):
        backend = self.backend_for(
            "terminus-2", template_task, model_name="openai/default"
        )
        config = backend.build_agent_config(
            template_task,
            ContainerInput(
                rollout_id="r1",
                chat_completions_url="http://t/v1",
                metadata={"harbor_model": "openai/override"},
            ),
        )
        assert config.model_name == "openai/override"


class TestGraderOutcome:
    """Failure precedence: the agent's own failure must stay primary.

    Phase is decided by timestamp against the verifier's start, never by
    class name; occurred_at is naive-local, timings aware-UTC, as in harbor.
    """

    # A naive-local base instant, as harbor's ExceptionInfo records it.
    BASE = datetime.now()

    def backend_for(self, template_task, tmp_path):
        return HarborBackendV2(
            orchestrator=TrialQueue(n_concurrent=1),
            tasks_dir=template_task,
            agent="terminus-2",
            trials_dir=tmp_path / "trials",
        )

    def verifier_span(self, start_offset_sec=30, duration_sec=30):
        from types import SimpleNamespace

        started = (self.BASE + timedelta(seconds=start_offset_sec)).astimezone(UTC)
        return SimpleNamespace(
            started_at=started,
            finished_at=started + timedelta(seconds=duration_sec),
        )

    def event_with(self, exception_info=None, verifier_result=None, verifier=None):
        from types import SimpleNamespace

        result = SimpleNamespace(
            exception_info=exception_info,
            verifier_result=verifier_result,
            started_at=None,
            finished_at=None,
            environment_setup=None,
            agent_setup=None,
            agent_execution=None,
            verifier=verifier,
        )
        return SimpleNamespace(result=result)

    def exception_at(self, offset_sec, exception_type="NonZeroAgentExitCodeError"):
        from types import SimpleNamespace

        return SimpleNamespace(
            exception_type=exception_type,
            exception_message="Command failed (exit 1): /app/agent_runner.py",
            exception_traceback="",
            occurred_at=self.BASE + timedelta(seconds=offset_sec),
        )

    async def test_agent_command_failure_stays_primary(self, template_task, tmp_path):
        """The verifier's missing-sample error must not replace the agent error."""
        from types import SimpleNamespace

        async def noop(result):
            pass

        backend = self.backend_for(template_task, tmp_path)
        event = self.event_with(
            exception_info=self.exception_at(0),
            verifier_result=SimpleNamespace(rewards={"reward": 0.0}),
            verifier=self.verifier_span(),
        )

        outcome = backend.grader_outcome(event, "r1", PendingTrial(noop, None))

        assert outcome.status == RolloutStatus.FAILURE
        assert outcome.err_category == RolloutErrorCategory.AGENT_ERROR
        assert "Command failed" in outcome.err_message

    async def test_classified_subclass_failure_stays_primary(
        self, template_task, tmp_path
    ):
        """The phase decision must not depend on the exception class name."""
        from types import SimpleNamespace

        async def noop(result):
            pass

        backend = self.backend_for(template_task, tmp_path)
        event = self.event_with(
            exception_info=self.exception_at(
                0, exception_type="AgentAuthenticationError"
            ),
            verifier_result=SimpleNamespace(rewards={"reward": 0.0}),
            verifier=self.verifier_span(),
        )

        outcome = backend.grader_outcome(event, "r1", PendingTrial(noop, None))

        assert outcome.err_category == RolloutErrorCategory.AGENT_ERROR

    async def test_missing_sample_after_clean_agent_is_validation_error(
        self, template_task, tmp_path
    ):
        from types import SimpleNamespace

        async def noop(result):
            pass

        backend = self.backend_for(template_task, tmp_path)
        event = self.event_with(
            verifier_result=SimpleNamespace(rewards={"reward": 1.0}),
            verifier=self.verifier_span(),
        )

        outcome = backend.grader_outcome(event, "r1", PendingTrial(noop, None))

        assert outcome.status == RolloutStatus.FAILURE
        assert outcome.err_category == RolloutErrorCategory.VALIDATION_ERROR
        assert "No sample to grade" in outcome.err_message

    async def test_post_verifier_exception_does_not_preempt_verifier(
        self, template_task, tmp_path
    ):
        """Teardown noise recorded after the verifier must not preempt it."""
        from types import SimpleNamespace

        async def noop(result):
            pass

        backend = self.backend_for(template_task, tmp_path)
        event = self.event_with(
            exception_info=self.exception_at(120),
            verifier_result=SimpleNamespace(rewards={"reward": 1.0}),
            verifier=self.verifier_span(start_offset_sec=30, duration_sec=30),
        )

        outcome = backend.grader_outcome(event, "r1", PendingTrial(noop, None))

        assert outcome.err_category == RolloutErrorCategory.VALIDATION_ERROR

    async def test_exception_without_verifier_run_stays_primary(
        self, template_task, tmp_path
    ):
        """When the verifier never started, any recorded exception precedes it."""

        async def noop(result):
            pass

        backend = self.backend_for(template_task, tmp_path)
        event = self.event_with(exception_info=self.exception_at(0))

        outcome = backend.grader_outcome(event, "r1", PendingTrial(noop, None))

        assert outcome.status == RolloutStatus.FAILURE
        assert outcome.err_category == RolloutErrorCategory.AGENT_ERROR
        assert "Command failed" in outcome.err_message


class TestTaskResolution:
    def backend_for(self, template_task):
        return HarborBackendV2(
            orchestrator=TrialQueue(n_concurrent=1),
            tasks_dir=template_task,
            agent="terminus-2",
        )

    async def test_template_mode_resolves_configured_dir(self, template_task):
        backend = self.backend_for(template_task)

        task = await backend.resolve_task(request_for())

        assert task.path == template_task

    async def test_fetched_task_instruction_replacement_warns(
        self, template_task, tmp_path, caplog
    ):
        """The warning keys on what is destroyed, not how the task was selected."""
        import logging
        from types import SimpleNamespace

        authored = tmp_path / "authored-task"
        (authored / "environment").mkdir(parents=True)
        (authored / "instruction.md").write_text("Task-owned instruction")

        async def failing_submit(config):
            raise RuntimeError("stop before harbor")

        orchestrator = SimpleNamespace(
            add_hook=lambda *args: None, submit=failing_submit
        )
        backend = HarborBackendV2(
            orchestrator=orchestrator,
            tasks_dir=template_task,
            agent="terminus-2",
        )

        async def fetch_local(ref, metadata):
            return HarborTask(authored)

        backend.fetch_task = fetch_local
        request = request_for(
            prompt=[{"role": "user", "content": "row prompt"}],
            metadata={"harbor_task": "./tasks/x"},
        )

        async def noop(result):
            pass

        with caplog.at_level(logging.WARNING):
            await backend.execute(request, noop)

        assert any(
            "template mode replaces the instruction.md" in record.getMessage()
            for record in caplog.records
        )

    async def test_configured_template_instruction_stays_silent(
        self, template_task, caplog
    ):
        """Replacing the template dir's own instruction.md is the intended flow."""
        import logging
        from types import SimpleNamespace

        (template_task / "instruction.md").write_text("template fallback")

        async def failing_submit(config):
            raise RuntimeError("stop before harbor")

        orchestrator = SimpleNamespace(
            add_hook=lambda *args: None, submit=failing_submit
        )
        backend = HarborBackendV2(
            orchestrator=orchestrator,
            tasks_dir=template_task,
            agent="terminus-2",
        )
        request = request_for(prompt=[{"role": "user", "content": "row prompt"}])

        async def noop(result):
            pass

        with caplog.at_level(logging.WARNING):
            await backend.execute(request, noop)

        assert not any(
            "template mode replaces the instruction.md" in record.getMessage()
            for record in caplog.records
        )


class TestDiagnostics:
    def result_with_timings(self):
        from datetime import UTC, datetime, timedelta
        from types import SimpleNamespace

        from harbor.models.trial.result import TimingInfo

        start = datetime(2026, 1, 1, tzinfo=UTC)

        def span(offset, seconds):
            return TimingInfo(
                started_at=start + timedelta(seconds=offset),
                finished_at=start + timedelta(seconds=offset + seconds),
            )

        return SimpleNamespace(
            started_at=start,
            finished_at=start + timedelta(seconds=100),
            environment_setup=span(0, 80),
            agent_setup=span(80, 10),
            agent_execution=span(90, 6),
            verifier=span(96, 4),
        )

    def test_trial_timings_reads_harbor_spans(self):
        from osmosis_ai.rollout.backend.harbor.diagnostics import trial_timings

        assert trial_timings(self.result_with_timings()) == {
            "environment_setup": 80.0,
            "agent_setup": 10.0,
            "agent": 6.0,
            "verifier": 4.0,
            "total": 100.0,
        }

    def test_failure_phase_is_furthest_span_reached(self):
        from osmosis_ai.rollout.backend.harbor.diagnostics import failure_phase

        result = self.result_with_timings()
        assert failure_phase(result) == "verifier"
        result.verifier = None
        result.agent_execution = None
        assert failure_phase(result) == "agent_setup"
        assert failure_phase(None) == "setup"

    def test_agent_phase_failure_compares_mixed_timezone_timestamps(self):
        """Mixed naive/aware timestamps must normalize, not raise TypeError."""
        from types import SimpleNamespace

        from osmosis_ai.rollout.backend.harbor.diagnostics import agent_phase_failure

        base = datetime.now()  # naive local, as ExceptionInfo records it
        verifier = SimpleNamespace(
            started_at=(base + timedelta(seconds=30)).astimezone(UTC),
            finished_at=(base + timedelta(seconds=60)).astimezone(UTC),
        )

        def result(occurred_at):
            return SimpleNamespace(
                exception_info=SimpleNamespace(occurred_at=occurred_at),
                verifier=verifier,
            )

        assert agent_phase_failure(result(base)) is not None
        assert agent_phase_failure(result(base + timedelta(seconds=120))) is None
        assert agent_phase_failure(None) is None
        assert (
            agent_phase_failure(SimpleNamespace(exception_info=None, verifier=None))
            is None
        )

    def test_redact_secrets_scrubs_keys_and_api_key(self):
        from osmosis_ai.rollout.backend.harbor.diagnostics import redact_secrets

        redacted = redact_secrets(
            {
                "llm_kwargs": {"api_key": "sk-1", "extra": ["Bearer sk-1", "safe"]},
                "model": "gpt",
                "session_token": "t0",
            },
            api_key="sk-1",
        )
        assert redacted == {
            "llm_kwargs": {"api_key": "[REDACTED]", "extra": ["[REDACTED]", "safe"]},
            "model": "gpt",
            "session_token": "[REDACTED]",
        }


class TestTaskRefs:
    def test_local_path_ref(self):
        from harbor.models.task.id import LocalTaskId

        from osmosis_ai.rollout.backend.harbor.tasks import parse_task_ref

        assert parse_task_ref("./tasks/t1", {}) == LocalTaskId(path=Path("./tasks/t1"))

    def test_package_ref_with_version(self):
        from harbor.models.task.id import PackageTaskId

        from osmosis_ai.rollout.backend.harbor.tasks import parse_task_ref

        assert parse_task_ref("laude/swe-bench@sha256:abc", {}) == PackageTaskId(
            org="laude", name="swe-bench", ref="sha256:abc"
        )

    def test_git_ref_uses_metadata(self):
        from harbor.models.task.id import GitTaskId

        from osmosis_ai.rollout.backend.harbor.tasks import parse_task_ref

        task_id = parse_task_ref(
            "tasks/t1",
            {"git_url": "https://github.com/org/tasks.git", "git_commit_id": "abc123"},
        )
        assert task_id == GitTaskId(
            git_url="https://github.com/org/tasks.git",
            git_commit_id="abc123",
            path=Path("tasks/t1"),
        )

    def test_bare_name_rejected(self):
        from osmosis_ai.rollout.backend.harbor.tasks import parse_task_ref

        with pytest.raises(ValueError, match="must be a local path"):
            parse_task_ref("not-a-ref", {})


class TestCancellation:
    def backend(self, bundle, template_task):
        return HarborBackendV2(
            orchestrator=TrialQueue(n_concurrent=1),
            tasks_dir=template_task,
            bundle=bundle,
        )

    async def hang(self):
        await asyncio.Event().wait()

    async def test_selectors_and_dispositions(self, bundle, template_task):
        backend = self.backend(bundle, template_task)

        async def noop(result):
            pass

        queued = PendingTrial(noop, None)
        queued.task = asyncio.create_task(self.hang())
        running = PendingTrial(noop, None)
        running.task = asyncio.create_task(self.hang())
        running.started = True
        backend.pending = {
            "job1-a": queued,
            "job1-b": running,
            "other": PendingTrial(noop, None),
        }

        assert backend.cancel_rollouts(ids=["missing"]) == {"missing": "not_found"}
        assert backend.cancel_rollouts(prefix="job1-") == {
            "job1-a": "cancelled_queued",
            "job1-b": "cancelled_running",
        }
        await asyncio.sleep(0)
        assert queued.task.cancelled()
        assert running.task.cancelled()
        # taskless entry: never submitted, nothing to cancel
        assert backend.cancel_rollouts(all=True)["other"] == "not_found"

    async def test_status_lifecycle(self, bundle, template_task):
        backend = self.backend(bundle, template_task)

        async def noop(result):
            pass

        assert backend.rollout_status("r1") is None

        pending = PendingTrial(noop, None)
        backend.pending["r1"] = pending
        assert backend.rollout_status("r1") == {"status": "queued"}
        pending.started = True
        assert backend.rollout_status("r1") == {"status": "running"}
        pending.grading = True
        assert backend.rollout_status("r1") == {"status": "grading"}

        backend.pending.pop("r1")
        backend.record_outcome("r1", RolloutStatus.SUCCESS, reward=1.0)
        assert backend.rollout_status("r1") == {
            "status": "success",
            "reward": 1.0,
            "err_message": None,
        }

        backend.finished.ttl_sec = -1.0
        backend.record_outcome("r2", RolloutStatus.FAILURE, err_message="boom")
        assert backend.rollout_status("r2") is None

    async def test_finished_task_is_not_found(self, bundle, template_task):
        backend = self.backend(bundle, template_task)

        async def noop(result):
            pass

        finished = PendingTrial(noop, None)
        finished.task = asyncio.create_task(asyncio.sleep(0))
        await finished.task
        backend.pending = {"done": finished}
        assert backend.cancel_rollouts(ids=["done"]) == {"done": "not_found"}


class TestPrewarm:
    def test_prewarm_config_is_install_only_without_credentials(
        self, bundle, template_task
    ):
        backend = HarborBackendV2(
            orchestrator=TrialQueue(n_concurrent=1),
            tasks_dir=template_task,
            bundle=bundle,
            agent_setup_timeout_sec=120,
        )
        config = backend.prewarm_trial_config(HarborTask(template_task))

        assert config.install_only is True
        assert config.verifier.disable is True
        assert config.trial_name.startswith("trial-prewarm-")
        assert config.agent.override_setup_timeout_sec == 120
        container_input = ContainerInput.read(
            Path(config.task.path) / "container_input.json"
        )
        assert container_input.api_key is None
        assert container_input.chat_completions_url in ("", None)

    async def test_dataset_prewarm_requires_task_ids(self, bundle, tmp_path):
        root = tmp_path / "dataset"
        root.mkdir()
        backend = HarborBackendV2(
            orchestrator=TrialQueue(n_concurrent=1),
            tasks_dir=root,
            task_mode=TaskMode.DATASET,
            bundle=bundle,
        )
        with pytest.raises(ValueError, match="requires task ids"):
            await backend.prewarm()


class FakeQueue:
    """A TrialQueue that fires hooks, then simulates harbor's secret scrub
    after the END hook and before submit() resolves."""

    def __init__(self, run=None):
        self.hooks = {}
        self.run = run

    def add_hook(self, event, hook):
        self.hooks.setdefault(event, []).append(hook)

    async def fire(self, event_name, event):
        from harbor.trial.hooks import TrialEvent

        for hook in self.hooks.get(TrialEvent(event_name), []):
            await hook(event)

    async def submit(self, config):
        assert self.run is not None, "FakeQueue.run not configured"
        return await self.run(self, config)


def trial_result(**overrides):
    from types import SimpleNamespace

    fields = {
        "exception_info": None,
        "verifier_result": None,
        "started_at": None,
        "finished_at": None,
        "environment_setup": None,
        "agent_setup": None,
        "agent_execution": None,
        "verifier": None,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


async def noop_callback(result):
    pass


class TestConfigValidation:
    def backend_for(self, template_task, **kwargs):
        return HarborBackendV2(
            orchestrator=TrialQueue(n_concurrent=1),
            tasks_dir=template_task,
            **kwargs,
        )

    def test_max_queue_depth_zero_rejected(self, template_task):
        """Depth 0 used to mean reject-everything (429 on an idle server)."""
        with pytest.raises(ValueError, match="max_queue_depth"):
            self.backend_for(template_task, agent="oracle", max_queue_depth=0)
        assert (
            self.backend_for(
                template_task, agent="oracle", max_queue_depth=1
            ).has_capacity()
            is True
        )

    def test_environment_config_not_aliased(self, template_task):
        from harbor.models.trial.config import (
            EnvironmentConfig as HarborEnvironmentConfig,
        )

        caller_config = HarborEnvironmentConfig()
        backend = self.backend_for(
            template_task, agent="oracle", environment_config=caller_config
        )
        caller_config.kwargs["poison"] = True

        assert backend.environment_config is not caller_config
        assert "poison" not in backend.environment_config.kwargs

    def test_missing_harbor_model_falls_back_to_default(self, template_task):
        backend = self.backend_for(template_task, agent="terminus-2")
        base = {"rollout_id": "r1", "chat_completions_url": "http://t/v1"}

        config = backend.build_agent_config(template_task, ContainerInput(**base))
        assert config.model_name == backend.model_name

    @pytest.mark.parametrize("agent", ["terminus-2", "mini-swe-agent"])
    def test_empty_endpoint_refused_for_wired_agents(self, template_task, agent):
        """An empty api_base sends the rollout credential to the public endpoint."""
        backend = self.backend_for(template_task, agent=agent)
        with pytest.raises(ValueError, match="no chat_completions_url"):
            backend.build_agent_config(
                template_task,
                ContainerInput(rollout_id="r1", api_key="rk-1"),
            )

    def test_oracle_needs_no_endpoint(self, template_task):
        backend = self.backend_for(template_task, agent="oracle")
        config = backend.build_agent_config(
            template_task, ContainerInput(rollout_id="r1")
        )
        assert config.name == "oracle"

    def test_env_wiring_sets_both_base_url_spellings(self, template_task):
        """Both OPENAI_* spellings must carry the rollout endpoint."""
        backend = self.backend_for(template_task, agent="mini-swe-agent")
        config = backend.build_agent_config(
            template_task,
            ContainerInput(
                rollout_id="r1", chat_completions_url="http://t/v1", api_key="k"
            ),
        )
        assert config.env["OPENAI_API_BASE"] == "http://t/v1"
        assert config.env["OPENAI_BASE_URL"] == "http://t/v1"


class TestDuplicateRolloutIds:
    async def test_duplicate_active_id_rejected(self, template_task):
        """A duplicate must not overwrite live pending state or staged files."""
        backend = HarborBackendV2(
            orchestrator=TrialQueue(n_concurrent=1),
            tasks_dir=template_task,
            agent="oracle",
        )
        original = PendingTrial(noop_callback, None)
        backend.pending["r1"] = original

        with pytest.raises(ValueError, match="already active"):
            await backend.execute(ExecutionRequest(id="r1", prompt=[]), noop_callback)

        assert backend.pending["r1"] is original


class TestArtifactLifecycle:
    def backend_for(self, template_task, tmp_path, queue, **kwargs):
        backend = HarborBackendV2(
            orchestrator=queue,
            tasks_dir=template_task,
            agent="terminus-2",
            trials_dir=tmp_path / "trials",
            **kwargs,
        )
        backend.rollouts_dir = tmp_path / "rollouts"
        backend.rollouts_dir.mkdir(parents=True, exist_ok=True)
        backend.artifact_root = tmp_path / "durable"
        return backend

    async def test_artifacts_relocate_only_after_harbor_scrub(
        self, template_task, tmp_path
    ):
        """The durable copy must be taken from the post-scrub tree."""
        from osmosis_ai.rollout.context import RolloutContext

        async def run(queue, config):
            artifacts = config.trials_dir / config.trial_name / "artifacts"
            artifacts.mkdir(parents=True)
            (artifacts / "out.txt").write_text("rk-secret")
            result = trial_result(
                verifier_result=__import__("types").SimpleNamespace(
                    rewards={"reward": 1.0}
                )
            )
            event = __import__("types").SimpleNamespace(config=config, result=result)
            await queue.fire("end", event)
            # Harbor's scrub runs here: after hooks, before submit() returns.
            (artifacts / "out.txt").write_text("[REDACTED]")
            return result

        queue = FakeQueue(run)
        backend = self.backend_for(template_task, tmp_path, queue)
        request = ExecutionRequest(id="r1", prompt=[{"role": "user", "content": "go"}])
        with RolloutContext(
            chat_completions_url="http://t/v1", api_key="rk-secret", rollout_id="r1"
        ):
            await backend.execute(request, noop_callback, noop_callback)

        relocated = tmp_path / "durable" / "r1" / "artifacts" / "out.txt"
        assert relocated.read_text() == "[REDACTED]"
        # The successful trial was moved only after the durable copy existed.
        assert not (tmp_path / "trials" / "trial-r1").exists()
        assert not (tmp_path / "rollouts" / "r1").exists()

    async def test_cancellation_cleans_rollout_and_trial_residue(
        self, template_task, tmp_path
    ):
        """Cancelled rollouts must not leave credential-bearing directories."""
        from osmosis_ai.rollout.context import RolloutContext

        started = asyncio.Event()

        async def run(queue, config):
            (config.trials_dir / config.trial_name).mkdir(parents=True)
            started.set()
            await asyncio.Event().wait()

        queue = FakeQueue(run)
        backend = self.backend_for(template_task, tmp_path, queue)
        request = ExecutionRequest(id="r1", prompt=[{"role": "user", "content": "go"}])

        async def execute():
            with RolloutContext(
                chat_completions_url="http://t/v1", api_key="rk-1", rollout_id="r1"
            ):
                await backend.execute(request, noop_callback)

        task = asyncio.create_task(execute())
        await started.wait()
        assert (tmp_path / "rollouts" / "r1").exists()

        assert backend.cancel_rollouts(ids=["r1"]) == {"r1": "cancelled_queued"}
        await task

        assert not (tmp_path / "rollouts" / "r1").exists()
        assert not (tmp_path / "trials" / "trial-r1").exists()
        assert "r1" not in backend.pending

    async def test_setup_failure_cleans_staging_and_reports(
        self, template_task, tmp_path
    ):
        from osmosis_ai.rollout.context import RolloutContext

        async def run(queue, config):
            raise RuntimeError("infra exploded")

        queue = FakeQueue(run)
        backend = self.backend_for(template_task, tmp_path, queue)
        request = ExecutionRequest(id="r1", prompt=[{"role": "user", "content": "go"}])
        delivered = []

        async def on_workflow_complete(result):
            delivered.append(result)

        with RolloutContext(
            chat_completions_url="http://t/v1", api_key="rk-1", rollout_id="r1"
        ):
            await backend.execute(request, on_workflow_complete)

        assert not (tmp_path / "rollouts" / "r1").exists()
        assert delivered and delivered[0].status == RolloutStatus.FAILURE

    async def test_failed_trial_keeps_trial_dir_for_debugging(
        self, template_task, tmp_path
    ):
        from types import SimpleNamespace

        from osmosis_ai.rollout.context import RolloutContext

        async def run(queue, config):
            artifacts = config.trials_dir / config.trial_name / "artifacts"
            artifacts.mkdir(parents=True)
            (artifacts / "log.txt").write_text("evidence")
            result = trial_result(
                exception_info=SimpleNamespace(
                    exception_type="AgentTimeoutError",
                    exception_message="too slow",
                    exception_traceback="",
                    occurred_at=None,
                )
            )
            event = SimpleNamespace(config=config, result=result)
            await queue.fire("end", event)
            return result

        queue = FakeQueue(run)
        backend = self.backend_for(template_task, tmp_path, queue)
        request = ExecutionRequest(id="r1", prompt=[{"role": "user", "content": "go"}])
        with RolloutContext(
            chat_completions_url="http://t/v1", api_key="rk-1", rollout_id="r1"
        ):
            await backend.execute(request, noop_callback)

        # Artifacts still copied for inspection; source retained on failure.
        assert (tmp_path / "durable" / "r1" / "artifacts" / "log.txt").exists()
        assert (tmp_path / "trials" / "trial-r1").exists()


class TestPrewarmIdentity:
    def test_prewarm_configs_are_credential_free(self, template_task, tmp_path):
        from osmosis_ai.rollout.context import RolloutContext

        for agent, check in (
            (
                "terminus-2",
                lambda cfg: (
                    "api_base" not in cfg.kwargs and "llm_kwargs" not in cfg.kwargs
                ),
            ),
            (
                "mini-swe-agent",
                lambda cfg: not any(k.startswith("OPENAI_") for k in cfg.env),
            ),
        ):
            backend = HarborBackendV2(
                orchestrator=TrialQueue(n_concurrent=1),
                tasks_dir=template_task,
                agent=agent,
                trials_dir=tmp_path / "trials",
            )
            # Prewarm must not pick up the ambient context's endpoint or key.
            with RolloutContext(
                chat_completions_url="http://t/v1", api_key="rk-1", rollout_id="x"
            ):
                config = backend.prewarm_trial_config(HarborTask(template_task))
            assert config.install_only is True
            assert check(config.agent), agent
            assert config.trial_name in backend.prewarm_trials

    async def test_prewarm_named_rollout_id_is_not_treated_as_prewarm(
        self, template_task
    ):
        """Dispatch must key on the registry, not a 'prewarm-' id pattern."""
        from types import SimpleNamespace

        backend = HarborBackendV2(
            orchestrator=TrialQueue(n_concurrent=1),
            tasks_dir=template_task,
            agent="oracle",
        )
        pending = PendingTrial(noop_callback, None)
        backend.pending["prewarm-abc"] = pending

        event = SimpleNamespace(
            config=SimpleNamespace(trial_name="trial-prewarm-abc"),
            result=trial_result(),
        )
        await backend.on_trial_end(event)

        # The rollout resolved normally instead of wedging execute() forever.
        assert "prewarm-abc" not in backend.pending
        assert pending.done.done()

    async def test_prewarm_cleanup_happens_at_call_site_post_scrub(
        self, template_task, tmp_path
    ):
        seen_dirs = {}

        async def run(queue, config):
            trial_dir = config.trials_dir / config.trial_name
            trial_dir.mkdir(parents=True, exist_ok=True)
            event = __import__("types").SimpleNamespace(
                config=config, result=trial_result()
            )
            await queue.fire("start", event)
            await queue.fire("end", event)
            # The END hook must leave the tree for harbor's scrub.
            seen_dirs[config.trial_name] = trial_dir.exists()
            return trial_result()

        queue = FakeQueue(run)
        backend = HarborBackendV2(
            orchestrator=queue,
            tasks_dir=template_task,
            agent="terminus-2",
            trials_dir=tmp_path / "trials",
        )
        backend.rollouts_dir = tmp_path / "rollouts"
        backend.rollouts_dir.mkdir(parents=True, exist_ok=True)

        await backend.prewarm()

        assert all(seen_dirs.values())
        assert backend.prewarm_trials == set()
        assert list((tmp_path / "trials").iterdir()) == []
        assert list((tmp_path / "rollouts").iterdir()) == []

    async def test_prewarm_keeps_directories_when_cleanup_disabled(
        self, template_task, tmp_path
    ):
        async def run(queue, config):
            (config.trials_dir / config.trial_name).mkdir(parents=True)
            return trial_result()

        queue = FakeQueue(run)
        backend = HarborBackendV2(
            orchestrator=queue,
            tasks_dir=template_task,
            agent="terminus-2",
            trials_dir=tmp_path / "trials",
            cleanup_successful_trials=False,
        )
        backend.rollouts_dir = tmp_path / "rollouts"
        backend.rollouts_dir.mkdir(parents=True, exist_ok=True)

        await backend.prewarm()

        assert list((tmp_path / "trials").iterdir()) != []


class TestContainerInputGating:
    def test_native_track_stages_no_container_input(self, template_task, tmp_path):
        backend = HarborBackendV2(
            orchestrator=TrialQueue(n_concurrent=1),
            tasks_dir=template_task,
            agent="mini-swe-agent",
        )
        backend.rollouts_dir = tmp_path / "rollouts"
        task_dir = backend.materialize_task(
            HarborTask(template_task),
            "r1",
            ContainerInput(
                rollout_id="r1",
                prompt=[{"role": "user", "content": "x"}],
                api_key="rk-1",
            ),
        )
        assert not (task_dir / "container_input.json").exists()

    def test_bundle_track_stages_container_input(self, bundle, template_task, tmp_path):
        backend = HarborBackendV2(
            orchestrator=TrialQueue(n_concurrent=1),
            tasks_dir=template_task,
            bundle=bundle,
        )
        backend.rollouts_dir = tmp_path / "rollouts"
        task_dir = backend.materialize_task(
            HarborTask(template_task),
            "r1",
            ContainerInput(rollout_id="r1", prompt=[{"role": "user", "content": "x"}]),
        )
        assert (task_dir / "container_input.json").exists()

    def test_native_with_grader_bundle_ships_input_only_in_tests(
        self, bundle, template_task, tmp_path
    ):
        backend = HarborBackendV2(
            orchestrator=TrialQueue(n_concurrent=1),
            tasks_dir=template_task,
            agent="terminus-2",
            grader="bench.solver:G",
            bundle=bundle,
        )
        backend.rollouts_dir = tmp_path / "rollouts"
        task_dir = backend.materialize_task(
            HarborTask(template_task),
            "r1",
            ContainerInput(rollout_id="r1", prompt=[{"role": "user", "content": "x"}]),
        )
        assert not (task_dir / "container_input.json").exists()
        assert (task_dir / "tests" / "container_input.json").exists()
