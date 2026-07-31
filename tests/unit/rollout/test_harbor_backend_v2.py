"""Bundle backend: contract round-trip, task materialization, trial config."""

import json

import pytest
from harbor.trial.queue import TrialQueue

from osmosis_ai.packaging import build_bundle
from osmosis_ai.rollout.backend.harbor.backend_v2 import HarborBackendV2
from osmosis_ai.rollout.backend.harbor.tasks import HarborTask, TaskMode
from osmosis_ai.rollout.container.files import ContainerInput, ContainerResult
from osmosis_ai.rollout.types.output import AgentWorkflowOutput, coerce_output
from osmosis_ai.rollout.types import ExecutionRequest, RolloutSample, RolloutStatus

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

        task_dir = HarborTask(template_task).materialize(tmp_path / "r1", container_input)

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
            ContainerInput(rollout_id="r1", prompt=prompt),
            grader_script="bench-grade",
        )
        assert "bench-grade" in (task_dir / "tests" / "test.sh").read_text()

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


class TestBundleBackend:
    @pytest.fixture
    def backend(self, bundle, template_task):
        return HarborBackendV2(
            orchestrator=TrialQueue(n_concurrent=1),
            tasks_dir=template_task,
            bundle=bundle,
        )

    def test_trial_config_wires_bundle_agent(self, backend, template_task, tmp_path):
        prompt = [{"role": "user", "content": "x"}]
        request = request_for(prompt)
        task_dir = backend.select_task(request).materialize(
            backend.rollouts_dir / request.id,
            backend.build_input(request),
            backend.bundle.grader_script,
        )
        config = backend.build_trial_config(task_dir, request)

        assert config.agent.import_path.endswith(":BundleAgent")
        assert config.agent.kwargs["agent_script"] == "bench-agent"
        assert config.agent.kwargs["bundle_path"] == str(backend.bundle.wheel)
        assert config.verifier.disable is False  # generated test.sh enables it

    def test_verifier_disabled_without_tests(self, bundle, template_task):
        backend = HarborBackendV2(
            orchestrator=TrialQueue(n_concurrent=1),
            tasks_dir=template_task,
            bundle=bundle,
        )
        prompt = [{"role": "user", "content": "x"}]
        request = request_for(prompt)
        task_dir = backend.select_task(request).materialize(
            backend.rollouts_dir / request.id, backend.build_input(request)
        )
        assert backend.build_trial_config(task_dir, request).verifier.disable is True

    def test_build_spec_carries_request_fields(self, backend):
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
