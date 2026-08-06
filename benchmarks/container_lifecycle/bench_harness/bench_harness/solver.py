"""Bench workflow: two chat calls against the rollout context's model URL."""

import httpx

from osmosis_ai.rollout import AgentWorkflow, AgentWorkflowContext, get_rollout_context


class BenchWorkflow(AgentWorkflow):
    async def run(self, ctx: AgentWorkflowContext) -> list[dict]:
        rollout_ctx = get_rollout_context()
        url = f"{rollout_ctx.chat_completions_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {rollout_ctx.api_key}"}
        messages = list(ctx.prompt)

        async with httpx.AsyncClient(timeout=60) as client:
            for _ in range(2):
                response = await client.post(
                    url,
                    json={"model": "bench", "messages": messages},
                    headers=headers,
                )
                response.raise_for_status()
                messages.append(response.json()["choices"][0]["message"])

        return messages
