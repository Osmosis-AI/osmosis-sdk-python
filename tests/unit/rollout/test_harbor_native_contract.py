"""Run Harbor's native agent adapters without launching a sandbox or model."""

import copy
import os
from unittest.mock import AsyncMock, patch

import pytest
import yaml
from harbor.agents.factory import AgentFactory
from harbor.models.agent.context import AgentContext

from osmosis_ai.rollout.backend.harbor.native_agents import (
    NATIVE_AGENTS,
    native_agent_config,
)


def mini_config(overrides, session="first"):
    return native_agent_config(
        "mini-swe-agent",
        NATIVE_AGENTS["mini-swe-agent"],
        "openai/student",
        f"https://trainer/sessions/{session}/v1",
        f"key-{session}",
        extra_kwargs=overrides,
    )


@pytest.mark.parametrize("config_source", ["none", "inline", "file"])
async def test_mini_preserves_chat_endpoint_and_reasoning_config(
    tmp_path, config_source
):
    custom = {
        "agent": {"step_limit": 20},
        "model": {
            "model_class": "litellm",
            "model_kwargs": {"temperature": 0.2, "reasoning_effort": "low"},
        },
    }
    overrides = {"reasoning_effort": "high", "max_tokens": 1234}
    if config_source == "inline":
        overrides["config"] = custom
    elif config_source == "file":
        config_file = tmp_path / "mini.yaml"
        config_file.write_text(yaml.safe_dump(custom))
        overrides["config_file"] = str(config_file)
    original = copy.deepcopy(overrides)
    expected = copy.deepcopy(custom) if config_source != "none" else {"model": {}}
    expected["model"].setdefault("model_kwargs", {})["reasoning_effort"] = "high"

    # Explicit per-session endpoint/key must win over unrelated host defaults.
    with patch.dict(
        os.environ,
        {
            "LITELLM_LOCAL_MODEL_COST_MAP": "true",
            "OPENAI_API_KEY": "host-key",
            "OPENAI_BASE_URL": "https://host.example/v1",
        },
        clear=True,
    ):
        for session in ("first", "second"):
            config = mini_config(overrides, session)
            # Harbor 0.22 predates preflight; newer releases validate their real
            # options model before constructing and running the native adapter.
            preflight = getattr(AgentFactory, "run_preflight", None)
            if preflight is not None:
                preflight(config)
            agent = AgentFactory.create_agent_from_config(
                config, tmp_path / f"logs-{session}"
            )
            agent.session_id = session
            agent.exec_as_agent = AsyncMock()
            await agent.run("Do nothing", object(), AgentContext())

            calls = agent.exec_as_agent.call_args_list
            command = calls[-1].kwargs["command"]
            env = calls[-1].kwargs["env"]
            assert "model.model_class=litellm_response" not in command
            assert "model.model_kwargs.reasoning.effort=" not in command
            config_command = next(
                call.kwargs["command"]
                for call in calls
                if "cat > '/tmp/mswea-config/custom.yaml'" in call.kwargs["command"]
            )
            written_yaml = config_command.split("\n", 2)[2].rsplit("\n", 2)[0]
            assert yaml.safe_load(written_yaml) == expected
            assert "model.model_kwargs.max_tokens=1234" in command
            assert "max_output_tokens" not in command
            assert f"extra_headers.X-Session-ID={session}" in command
            assert env["OPENAI_BASE_URL"] == f"https://trainer/sessions/{session}/v1"
            assert env["OPENAI_API_BASE"] == env["OPENAI_BASE_URL"]
            assert env["OPENAI_API_KEY"] == f"key-{session}"
            assert f"key-{session}" not in command
            assert overrides == original
            config.kwargs["config"]["model"]["model_kwargs"]["temperature"] = 99


@pytest.mark.parametrize("effort", [None, ""])
def test_mini_does_not_read_config_file_without_reasoning_override(tmp_path, effort):
    overrides = {
        "config_file": str(tmp_path / "not-created.yaml"),
        "reasoning_effort": effort,
    }
    assert mini_config(overrides).kwargs == overrides


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"config": {}, "config_file": "unused.yaml"}, "mutually exclusive"),
        ({"config": []}, "config.*mapping"),
        ({"config": {"model": []}}, "model.*mapping"),
        ({"config": {"model": {"model_kwargs": []}}}, "model_kwargs.*mapping"),
    ],
)
def test_mini_rejects_invalid_reasoning_config(overrides, message):
    with pytest.raises(ValueError, match=message):
        mini_config({"reasoning_effort": "high", **overrides})
