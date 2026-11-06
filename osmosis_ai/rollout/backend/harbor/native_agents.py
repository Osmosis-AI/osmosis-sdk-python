"""Registered native Harbor agents and how each receives the model endpoint."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from harbor.models.trial.config import AgentConfig as HarborAgentConfig


@dataclass(frozen=True)
class NativeAgentBinding:
    wiring: Literal["env", "kwargs", "opencode", "pi", "none"]
    trainable: bool = True
    env: dict[str, str] = field(default_factory=dict)
    kwargs: dict[str, Any] = field(default_factory=dict)


NATIVE_AGENTS: dict[str, NativeAgentBinding] = {
    "terminus-2": NativeAgentBinding(
        wiring="kwargs",
        # Summarization rewrites the running context, which forks the token
        # trajectory RL training needs to stay append-only.
        kwargs={"enable_summarize": False},
    ),
    "mini-swe-agent": NativeAgentBinding(
        wiring="env",
        env={"MSWEA_COST_TRACKING": "ignore_errors"},
    ),
    "opencode": NativeAgentBinding(wiring="opencode"),
    "pi": NativeAgentBinding(wiring="pi"),
    # Runs the task's reference solution with no model traffic; validates
    # datasets and verifiers, never produces training data.
    "oracle": NativeAgentBinding(wiring="none", trainable=False),
}


_PREBAKED_OPENCODE_BIN_DIR = "/opt/osmosis-agent/bin"
_MUSL_CHECK = "ldd --version 2>&1 | grep -qi musl || [ -f /etc/alpine-release ]"
_OPENCODE_PATCHED = False
_PI_PATCHED = False


def _convert_pi_events_to_trajectory(
    self: Any,
    lines: Iterable[str | bytes],
) -> Any:
    import json

    from harbor.models.trajectories import (
        Agent,
        FinalMetrics,
        Metrics,
        Observation,
        ObservationResult,
        Step,
        ToolCall,
        Trajectory,
    )

    def _int_val(value: Any) -> int:
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    def _metrics(usage: Any) -> Any:
        if not isinstance(usage, dict):
            return None
        inp = _int_val(usage.get("input"))
        out = _int_val(usage.get("output"))
        cr = _int_val(usage.get("cacheRead"))
        cw = _int_val(usage.get("cacheWrite"))
        reas = _int_val(usage.get("reasoning"))
        tot = _int_val(usage.get("totalTokens"))
        if not (inp or out or cr or cw or reas or tot):
            return None
        cost = usage.get("cost")
        cost_total = cost.get("total") if isinstance(cost, dict) else None
        cost_usd = (
            float(cost_total)
            if isinstance(cost_total, int | float) and cost_total > 0
            else None
        )
        return Metrics(
            prompt_tokens=inp + cr + cw,
            completion_tokens=out,
            cached_tokens=cr,
            cost_usd=cost_usd,
            extra={
                "cache_write_tokens": cw,
                "reasoning_tokens": reas,
                "pi_total_tokens": tot,
            },
        )

    def _content_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        return "\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ).strip()

    session_id = "unknown"
    default_model_name = self.model_name
    seen_message_type: str | None = None
    steps: list[Step] = []
    call_steps: dict[str, Step] = {}
    total_prompt = 0
    total_completion = 0
    total_cached = 0
    total_cost = 0.0

    for raw_line in lines:
        try:
            event = json.loads(raw_line)
        except (ValueError, UnicodeDecodeError):
            continue
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if event_type == "session":
            sid = event.get("id")
            if session_id == "unknown" and isinstance(sid, str) and sid:
                session_id = sid
            continue
        if event_type == "model_change":
            prov = event.get("provider")
            mid = event.get("modelId")
            if isinstance(prov, str) and isinstance(mid, str):
                default_model_name = f"{prov}/{mid}"
            continue
        if event_type not in {"message", "message_end"}:
            continue
        seen_message_type = seen_message_type or event_type
        if event_type != seen_message_type:
            continue
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "user":
            steps.append(
                Step(
                    step_id=len(steps) + 1,
                    timestamp=event.get("timestamp")
                    if isinstance(event.get("timestamp"), str)
                    else None,
                    source="user",
                    message=_content_text(message.get("content")),
                )
            )
            continue
        if role == "assistant":
            text_blocks: list[str] = []
            reasoning_blocks: list[str] = []
            tool_calls: list[ToolCall] = []
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "text" and isinstance(block.get("text"), str):
                        text_blocks.append(block["text"])
                    elif btype == "thinking" and isinstance(block.get("thinking"), str):
                        reasoning_blocks.append(block["thinking"])
                    elif btype == "toolCall":
                        tc_id = block.get("id")
                        fn_name = block.get("name")
                        if isinstance(tc_id, str) and isinstance(fn_name, str):
                            args = block.get("arguments")
                            tool_calls.append(
                                ToolCall(
                                    tool_call_id=tc_id,
                                    function_name=fn_name,
                                    arguments=args if isinstance(args, dict) else {},
                                )
                            )
            if (
                message.get("stopReason") in {"aborted", "error"}
                and not text_blocks
                and not reasoning_blocks
                and not tool_calls
            ):
                continue
            m = _metrics(message.get("usage"))
            if m is not None:
                total_prompt += m.prompt_tokens or 0
                total_completion += m.completion_tokens or 0
                total_cached += m.cached_tokens or 0
                total_cost += m.cost_usd or 0.0
            step = Step(
                step_id=len(steps) + 1,
                timestamp=event.get("timestamp")
                if isinstance(event.get("timestamp"), str)
                else None,
                source="agent",
                model_name=default_model_name,
                message="\n".join(text_blocks).strip(),
                reasoning_content="\n".join(reasoning_blocks).strip() or None,
                tool_calls=tool_calls or None,
                observation=Observation(results=[]) if tool_calls else None,
                metrics=m,
                llm_call_count=1,
            )
            steps.append(step)
            for tc in tool_calls:
                call_steps[tc.tool_call_id] = step
            continue
        if role == "toolResult":
            tc_id = message.get("toolCallId")
            if not isinstance(tc_id, str):
                continue
            step = call_steps.get(tc_id)
            if step is None:
                continue
            if step.observation is None:
                step.observation = Observation(results=[])
            extra: dict[str, Any] = {
                "tool_name": message.get("toolName"),
                "is_error": bool(message.get("isError", False)),
            }
            details = message.get("details")
            if isinstance(details, dict):
                extra["details"] = details
                if isinstance(details.get("results"), list):
                    for sub_res in details["results"]:
                        if not isinstance(sub_res, dict):
                            continue
                        sub_m = _metrics(sub_res.get("usage"))
                        if sub_m is not None:
                            total_prompt += sub_m.prompt_tokens or 0
                            total_completion += sub_m.completion_tokens or 0
                            total_cached += sub_m.cached_tokens or 0
                            total_cost += sub_m.cost_usd or 0.0
            step.observation.results.append(
                ObservationResult(
                    source_call_id=tc_id,
                    content=_content_text(message.get("content")),
                    extra=extra,
                )
            )

    if not steps:
        return None
    return Trajectory(
        schema_version="ATIF-v1.7",
        session_id=session_id,
        agent=Agent(
            name=self.name(),
            version=self.version() or "unknown",
            model_name=default_model_name,
        ),
        steps=steps,
        final_metrics=FinalMetrics(
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            total_cached_tokens=total_cached,
            total_cost_usd=total_cost if total_cost > 0 else None,
            total_steps=len(steps),
        ),
    )


_PI_SUBAGENT_AGENTS: dict[str, str] = {
    name: (
        f"---\nname: {name}\n"
        "description: General-purpose subagent with full capabilities and isolated context\n"
        "tools: read, bash, edit, write, grep, find, ls\n---\n\n"
        f"You are a {name} subagent running in an isolated context window. "
        "Complete the delegated task autonomously using the available tools and return the result concisely.\n"
    )
    for name in ("general", "worker")
}


def _adapt_pi_subagent_instruction(instruction: str) -> str:
    if not isinstance(instruction, str):
        return instruction
    updated = instruction.replace(
        "Execute the required plan using real OpenCode subagents.",
        "Execute the required plan using real Pi subagents.",
    )
    return updated.replace(
        'Your first action must be native `task` calls with `subagent_type="general"`, '
        "one fresh child per plan step. Omit `task_id`; never reuse a child.",
        'Your first action must be native `subagent` tool calls with `agent="general"` '
        '(using single mode `{"agent": "general", "task": "<child prompt>"}` or parallel mode '
        '`{"tasks": [{"agent": "general", "task": "<child prompt>"}, ...]}`), '
        "one fresh child per plan step. Never reuse a child.",
    )


def _ensure_pi_support() -> None:
    global _PI_PATCHED
    if _PI_PATCHED:
        return
    import contextlib
    import json
    import shlex
    from types import SimpleNamespace

    from harbor.agents.installed.pi import Pi
    from harbor.utils.trajectory_utils import format_trajectory_json

    has_native_custom_models = hasattr(Pi, "_build_custom_models_json")
    if not hasattr(Pi, "_upload_config_text"):

        async def _fallback_upload_config_text(
            self: Any,
            environment: Any,
            *,
            content: str,
            remote_path: str,
            filename: str,
        ) -> None:
            _ = filename
            await self.exec_as_agent(
                environment,
                command=(
                    f"mkdir -p $(dirname {shlex.quote(remote_path)}) && "
                    f"printf '%s' {shlex.quote(content)} > {shlex.quote(remote_path)}"
                ),
            )

        Pi._upload_config_text = _fallback_upload_config_text

    if has_native_custom_models:
        orig_build_models = Pi._build_custom_models_json
        orig_write_models = Pi._write_custom_models_json
    else:
        orig_init = Pi.__init__

        def _init_pi(
            self: Any, *args: Any, model_api: str | None = None, **kwargs: Any
        ) -> None:
            if model_api is not None and not isinstance(model_api, str):
                raise TypeError("model_api must be a string")
            orig_init(self, *args, **kwargs)
            self._model_api = model_api

        Pi.__init__ = _init_pi

        def orig_build_models(
            self: Any, access: Any, model_id: str
        ) -> dict[str, Any] | None:
            endpoint = getattr(access, "configured_base_url", None)
            if not endpoint:
                return None
            model_api = getattr(self, "_model_api", None) or "openai-completions"
            return {
                "providers": {
                    "harbor-endpoint": {
                        "baseUrl": endpoint,
                        "apiKey": "$OPENAI_API_KEY",
                        "api": model_api,
                        "models": [{"id": model_id}],
                    }
                }
            }

        async def orig_write_models(
            self: Any, environment: Any, models_json: dict[str, Any]
        ) -> None:
            await self.exec_as_agent(
                environment,
                command="mkdir -p /tmp/harbor-pi-agent && chmod 700 /tmp/harbor-pi-agent",
            )
            await self._upload_config_text(
                environment,
                content=json.dumps(models_json, indent=2) + "\n",
                remote_path="/tmp/harbor-pi-agent/models.json",
                filename="models.json",
            )
            await self.exec_as_agent(
                environment,
                command="chmod 600 /tmp/harbor-pi-agent/models.json",
            )

    orig_run = Pi.run

    def _build_custom_models_json(
        self: Any, access: Any, model_id: str
    ) -> dict[str, Any] | None:
        models_json = orig_build_models(self, access, model_id)
        if isinstance(models_json, dict):
            providers = models_json.get("providers")
            if isinstance(providers, dict):
                provider_cfg = providers.get("harbor-endpoint")
                if isinstance(provider_cfg, dict):
                    compat = provider_cfg.setdefault("compat", {})
                    if isinstance(compat, dict):
                        compat.setdefault("supportsStore", False)
                        compat.setdefault("supportsDeveloperRole", False)
                        compat.setdefault("maxTokensField", "max_tokens")
        return models_json

    async def _write_custom_models_json(
        self: Any, environment: Any, models_json: dict[str, Any]
    ) -> None:
        await orig_write_models(self, environment, models_json)
        await self._upload_config_text(
            environment,
            content=json.dumps({"compaction": {"enabled": False}}, indent=2) + "\n",
            remote_path="/tmp/harbor-pi-agent/settings.json",
            filename="settings.json",
        )
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                "[ -f ~/.nvm/nvm.sh ] && . ~/.nvm/nvm.sh; "
                "mkdir -p /logs/agent/pi/sessions /tmp/harbor-pi-agent/extensions/subagent /tmp/harbor-pi-agent/agents; "
                'npm_root="$(npm root -g)"; '
                'for pkg in "@earendil-works/pi-coding-agent" "@mariozechner/pi-coding-agent"; do '
                'sub_dir="$npm_root/$pkg/examples/extensions/subagent"; '
                'if [ -f "$sub_dir/index.ts" ] && [ -f "$sub_dir/agents.ts" ]; then '
                'cp "$sub_dir/index.ts" /tmp/harbor-pi-agent/extensions/subagent/index.ts && '
                'cp "$sub_dir/agents.ts" /tmp/harbor-pi-agent/extensions/subagent/agents.ts; '
                "break; "
                "fi; "
                "done; "
                "[ -f /tmp/harbor-pi-agent/extensions/subagent/index.ts ] && "
                "[ -f /tmp/harbor-pi-agent/extensions/subagent/agents.ts ]"
            ),
        )
        for agent_name, agent_md in _PI_SUBAGENT_AGENTS.items():
            await self._upload_config_text(
                environment,
                content=agent_md,
                remote_path=f"/tmp/harbor-pi-agent/agents/{agent_name}.md",
                filename=f"{agent_name}.md",
            )

    async def _run_pi(
        self: Any,
        instruction: str,
        environment: Any,
        context: Any,
    ) -> None:
        adapted = _adapt_pi_subagent_instruction(instruction)
        if not has_native_custom_models and self.model_name and "/" in self.model_name:
            endpoint = self._get_env("OPENAI_BASE_URL") or self._get_env(
                "OPENAI_API_BASE"
            )
            if endpoint:
                _, model_id = self.model_name.split("/", 1)
                access = SimpleNamespace(
                    configured_base_url=endpoint,
                    api_key=self._get_env("OPENAI_API_KEY"),
                    env=dict(self._extra_env),
                )
                models_json = self._build_custom_models_json(access, model_id)
                if models_json is not None:
                    await self._write_custom_models_json(environment, models_json)
                escaped_instruction = shlex.quote(
                    self.render_instruction(adapted)
                    if hasattr(self, "render_instruction")
                    else adapted
                )
                cli_flags = self.build_cli_flags()
                if cli_flags:
                    cli_flags += " "
                resume_flag = "--continue " if getattr(self, "_resume", False) else ""
                skills_command = self._build_register_skills_command()
                if skills_command:
                    await self.exec_as_agent(environment, command=skills_command)
                env = {
                    k: v
                    for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE")
                    if (v := self._get_env(k)) is not None
                }
                await self.exec_as_agent(
                    environment,
                    command=(
                        ". ~/.nvm/nvm.sh; "
                        "PI_CODING_AGENT_DIR=/tmp/harbor-pi-agent "
                        "pi --print --mode json "
                        "--session-dir /logs/agent/pi/sessions "
                        f"{resume_flag}"
                        f"--provider harbor-endpoint --model {model_id} "
                        f"{cli_flags}"
                        f"{escaped_instruction} "
                        f'2>&1 </dev/null | grep -v \'"type":"message_update"\' | stdbuf -oL tee /logs/agent/{self._OUTPUT_FILENAME}'
                    ),
                    env=env,
                )
                return
        await orig_run(self, adapted, environment, context)

    Pi._build_custom_models_json = _build_custom_models_json
    Pi._write_custom_models_json = _write_custom_models_json
    Pi.run = _run_pi

    orig_populate = Pi.populate_context_post_run

    def populate_context_post_run(self: Any, context: Any) -> None:
        orig_populate(self, context)
        trajectory_path = self.logs_dir / "trajectory.json"
        if trajectory_path.is_file():
            return
        session_dir = self.logs_dir / "pi" / "sessions"
        lines: list[str] = []
        try:
            session_files = (
                sorted(
                    session_dir.glob("*.jsonl"),
                    key=lambda p: (p.stat().st_mtime_ns, p.name),
                )
                if session_dir.is_dir()
                else []
            )
            if session_files:
                for sf in session_files:
                    lines.extend(sf.read_text().splitlines())
            elif (self.logs_dir / "pi.txt").is_file():
                lines = (self.logs_dir / "pi.txt").read_text().splitlines()
        except OSError:
            return
        trajectory = _convert_pi_events_to_trajectory(self, lines)
        if trajectory is not None:
            with contextlib.suppress(OSError):
                trajectory_path.write_text(
                    format_trajectory_json(trajectory.to_json_dict())
                )

    Pi.populate_context_post_run = populate_context_post_run

    _PI_PATCHED = True


def _ensure_opencode_prebaked_support() -> None:
    global _OPENCODE_PATCHED
    if _OPENCODE_PATCHED:
        return
    from harbor.agents.installed.opencode import OpenCode

    orig_install = OpenCode.install

    def get_version_command(self: Any) -> str | None:
        return (
            f'export PATH="{_PREBAKED_OPENCODE_BIN_DIR}:$PATH"; '
            "[ -f ~/.nvm/nvm.sh ] && . ~/.nvm/nvm.sh; opencode --version"
        )

    async def install(self: Any, environment: Any) -> None:
        version_cmd = self.get_version_command()
        if version_cmd:
            probe_cmd = (
                f"command -v stdbuf >/dev/null 2>&1 && {{ {version_cmd}; }}"
                if self._version is not None
                else (
                    f"[ -x {_PREBAKED_OPENCODE_BIN_DIR}/opencode ] && "
                    f"command -v stdbuf >/dev/null 2>&1 && {{ {version_cmd}; }}"
                )
            )
            probe = await environment.exec(command=probe_cmd)
            if (
                probe.return_code == 0
                and probe.stdout
                and (
                    self.parse_version(probe.stdout) == self._version
                    if self._version is not None
                    else bool(self.parse_version(probe.stdout))
                )
            ):
                return

        orig_ensure = getattr(self, "ensure_system_dependencies", None)
        if orig_ensure is not None:
            musl_check = await environment.exec(command=_MUSL_CHECK)
            is_musl = musl_check.return_code == 0

            async def ensure_filtered(env: Any, deps: tuple[str, ...]) -> None:
                filtered = (
                    deps
                    if is_musl
                    else tuple(d for d in deps if d not in {"nodejs", "npm"})
                )
                await orig_ensure(env, filtered)

            self.ensure_system_dependencies = ensure_filtered
        try:
            await orig_install(self, environment)
        finally:
            if orig_ensure is not None:
                self.ensure_system_dependencies = orig_ensure

    OpenCode.get_version_command = get_version_command
    OpenCode.install = install
    _OPENCODE_PATCHED = True


def native_prewarm_agent_config(
    name: str,
    binding: NativeAgentBinding,
    model_name: str,
    extra_kwargs: dict[str, Any] | None = None,
) -> HarborAgentConfig:
    """Setup-only config: installs the agent with no endpoint or credentials."""
    if binding.wiring == "opencode":
        _ensure_opencode_prebaked_support()
    elif binding.wiring == "pi":
        _ensure_pi_support()
    kwargs = copy.deepcopy({**binding.kwargs, **(extra_kwargs or {})})
    kwargs.pop("_osmosis_preview_path", None)
    return HarborAgentConfig(
        name=name,
        model_name=model_name,
        env=dict(binding.env),
        kwargs=kwargs,
    )


def native_binding(agent: Any) -> NativeAgentBinding | None:
    """The binding for a registered native agent name; None for workflow agents."""
    if not isinstance(agent, str) or ":" in agent:
        return None
    binding = NATIVE_AGENTS.get(agent)
    if binding is None:
        raise ValueError(
            f"unknown native agent {agent!r}; registered: {sorted(NATIVE_AGENTS)}"
        )
    return binding


def native_agent_config(
    name: str,
    binding: NativeAgentBinding,
    model_name: str,
    url: str,
    api_key: str,
    extra_kwargs: dict[str, Any] | None = None,
    *,
    preview_path: Path | None = None,
) -> HarborAgentConfig:
    kwargs = copy.deepcopy({**binding.kwargs, **(extra_kwargs or {})})
    kwargs.pop("_osmosis_preview_path", None)
    if binding.wiring == "none":
        return HarborAgentConfig(
            name=name, model_name=model_name, env=dict(binding.env), kwargs=kwargs
        )
    config_name: str | None = name
    import_path = None
    if preview_path is not None:
        wrapper = {
            "terminus-2": "_PreviewTerminus2",
            "mini-swe-agent": "_PreviewMiniSweAgent",
            "opencode": "_PreviewOpenCode",
            "pi": "_PreviewPi",
        }[name]
        config_name = None
        import_path = f"osmosis_ai.rollout.backend.harbor.preview_agents:{wrapper}"
        kwargs["_osmosis_preview_path"] = str(preview_path)
    if binding.wiring == "pi":
        _ensure_pi_support()
        if not isinstance(model_name, str):
            raise ValueError("Pi model_name must have the form provider/model")
        provider, separator, model_id = model_name.partition("/")
        if not separator or not provider or not model_id:
            raise ValueError("Pi model_name must have the form provider/model")
        kwargs.setdefault("model_api", "openai-completions")
        return HarborAgentConfig(
            name=config_name,
            import_path=import_path,
            model_name=f"openai/{model_id}",
            env={
                **binding.env,
                "OPENAI_API_BASE": url,
                "OPENAI_BASE_URL": url,
                "OPENAI_API_KEY": api_key,
            },
            kwargs=kwargs,
        )
    if binding.wiring == "opencode":
        _ensure_opencode_prebaked_support()
        if not isinstance(model_name, str):
            raise ValueError("OpenCode model_name must have the form provider/model")
        provider, separator, model_id = model_name.partition("/")
        if not separator or not provider or not model_id:
            raise ValueError("OpenCode model_name must have the form provider/model")
        config = kwargs.setdefault("opencode_config", {})
        # Built-in provider IDs have OpenCode-specific loaders. In particular,
        # "openai" selects sdk.responses even when npm is overridden. A private
        # provider ID keeps the session on the chat-completions protocol.
        session_provider = "osmosis-rollout"
        providers = config.setdefault("provider", {})
        provider_config = providers.pop(provider, {})
        suffix = 0
        while session_provider in providers:
            suffix += 1
            session_provider = f"osmosis-rollout-{suffix}"
        providers[session_provider] = provider_config
        provider_config["npm"] = "@ai-sdk/openai-compatible"
        provider_config.setdefault("options", {}).update(
            {"baseURL": url, "apiKey": "{env:OPENAI_API_KEY}"}
        )
        provider_config.setdefault("models", {}).setdefault(model_id, {})
        # Compaction rewrites history and breaks the training token trajectory.
        config.setdefault("compaction", {}).update({"auto": False, "prune": False})
        return HarborAgentConfig(
            name=config_name,
            import_path=import_path,
            model_name=f"{session_provider}/{model_id}",
            env={
                **binding.env,
                "OPENAI_API_BASE": url,
                "OPENAI_BASE_URL": url,
                "OPENAI_API_KEY": api_key,
            },
            kwargs=kwargs,
        )
    if binding.wiring == "env":
        if name == "mini-swe-agent" and (effort := kwargs.get("reasoning_effort")):
            # Harbor's top-level effort switches openai/* to Responses even for
            # a chat endpoint. The portable mini config preserves that endpoint.
            if not isinstance(effort, str):
                raise ValueError("mini-swe-agent reasoning_effort must be a string")
            config = kwargs.get("config")
            config_file = kwargs.get("config_file")
            if config is not None and config_file is not None:
                raise ValueError("'config' and 'config_file' are mutually exclusive")
            if config_file:
                import yaml

                config = yaml.safe_load(Path(config_file).read_text())
            if config is None:
                config = {}
            if not isinstance(config, dict):
                raise ValueError("mini-swe-agent config must be a mapping")
            kwargs["config"] = config
            kwargs.pop("config_file", None)
            for section in ("model", "model_kwargs"):
                config = config.setdefault(section, {})
                if not isinstance(config, dict):
                    raise ValueError(f"mini-swe-agent {section} must be a mapping")
            config["reasoning_effort"] = kwargs.pop("reasoning_effort")
        # mini-swe-agent reads OPENAI_BASE_URL before OPENAI_API_BASE; set both
        # so a host-level value can never outrank the rollout endpoint.
        env = {
            **binding.env,
            "OPENAI_API_BASE": url,
            "OPENAI_BASE_URL": url,
            "OPENAI_API_KEY": api_key,
        }
        return HarborAgentConfig(
            name=config_name,
            import_path=import_path,
            model_name=model_name,
            env=env,
            kwargs=kwargs,
        )
    # Kwargs-wired agents (terminus-2) silently drop a top-level api_key, so
    # the rollout key rides inside llm_kwargs. Endpoint wiring wins over user
    # kwargs: the rollout URL is not optional.
    llm_kwargs = {**kwargs.pop("llm_kwargs", {}), "api_key": api_key}
    return HarborAgentConfig(
        name=config_name,
        import_path=import_path,
        model_name=model_name,
        kwargs={**kwargs, "api_base": url, "llm_kwargs": llm_kwargs},
    )
