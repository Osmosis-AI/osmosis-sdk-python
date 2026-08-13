"""Frozen eval-proxy wire contract against real agent integrations."""

from __future__ import annotations

import pytest

from osmosis_ai.rollout.context import RolloutContext
from osmosis_ai.rollout.controller.listener import LocalhostUvicornServer
from osmosis_ai.rollout.controller.proxy_client import (
    EVAL_PROXY_INTEGRATION_MODEL,
    EVAL_PROXY_WIRE_MODEL,
    EvalProxyClient,
    create_eval_proxy_stub_app,
)

ROLLOUT_ID = "e" * 32
MODEL_PATH = "openai/gpt-4.1-mini"

pytest.importorskip("agents")
pytest.importorskip("litellm")
pytest.importorskip("strands")


@pytest.fixture
async def eval_proxy_stub():
    app = create_eval_proxy_stub_app()
    async with LocalhostUvicornServer(app) as server:
        client = EvalProxyClient(base_url=server.base_url, auth_token="platform-token")
        session = await client.create_session(
            rollout_id=ROLLOUT_ID,
            model_path=MODEL_PATH,
        )
        try:
            yield app, session
        finally:
            await client.aclose()


async def test_openai_agents_sends_frozen_wire_contract(eval_proxy_stub) -> None:
    from agents import RunConfig, Runner

    from osmosis_ai.rollout.integrations.agents.openai_agents import (
        OsmosisAgent,
        OsmosisMemorySession,
        OsmosisRolloutModel,
    )

    app, session = eval_proxy_stub
    ctx = RolloutContext(
        chat_completions_url=session.api_base_url,
        api_key=session.token,
        rollout_id=session.rollout_id,
    )
    with ctx:
        memory = OsmosisMemorySession()
        agent = OsmosisAgent(name="main", model=OsmosisRolloutModel())
        assert agent.model.model == EVAL_PROXY_INTEGRATION_MODEL
        await Runner.run(
            agent,
            "hello",
            session=memory,
            run_config=RunConfig(tracing_disabled=True),
        )

    recorded = app.state.chat_requests[-1]
    assert recorded["path"] == f"/v1/eval-sessions/{ROLLOUT_ID}/chat/completions"
    body = recorded["body"]
    assert body["model"] == EVAL_PROXY_WIRE_MODEL
    assert body["stream"] is True
    assert "stream_options" not in body


async def test_strands_sends_frozen_wire_contract(eval_proxy_stub) -> None:
    from osmosis_ai.rollout.integrations.agents.strands import (
        OsmosisRolloutModel,
        OsmosisStrandsAgent,
    )

    app, session = eval_proxy_stub
    ctx = RolloutContext(
        chat_completions_url=session.api_base_url,
        api_key=session.token,
        rollout_id=session.rollout_id,
    )
    with ctx:
        agent = OsmosisStrandsAgent(name="solver", model=OsmosisRolloutModel())
        await agent.invoke_async("hello")

    recorded = app.state.chat_requests[-1]
    assert recorded["path"] == f"/v1/eval-sessions/{ROLLOUT_ID}/chat/completions"
    body = recorded["body"]
    assert body["model"] == EVAL_PROXY_WIRE_MODEL
    assert body["stream"] is True
    stream_options = body.get("stream_options")
    if stream_options is not None:
        assert stream_options.get("include_usage") is True
    else:
        pytest.fail("Strands must send include_usage=true when stream_options is set")
