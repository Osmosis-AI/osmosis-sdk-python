from __future__ import annotations

from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    from openai import BadRequestError, OpenAI, OpenAIError  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]
    BadRequestError = None  # type: ignore[assignment]
    OpenAIError = None  # type: ignore[assignment]

from ..rubric_types import ProviderRequestError, RewardRubricRunResult
from .base import DEFAULT_REQUEST_TIMEOUT_SECONDS, ProviderRequest, RubricProvider
from .shared import dump_model, reward_json_schema, reward_schema_definition, sanitize_json

# Families that generally require "max_completion_tokens" on chat.completions in SDK v2.
FAMILIES_USE_MAX_COMPLETION = ("gpt-5", "gpt-4.1", "gpt-4o", "o3", "o4")

# ---- Tunables for safer defaults ----
# Higher token budget avoids "reasoning-only consumed budget" on gpt-5.
SAFE_DEFAULT_MAX_OUTPUT = 1536  # you can raise to 2048 if you prefer
SAFE_MAX_OUTPUT_CAP = 4096      # hard safety cap to avoid unbounded growth
SAFE_BUMP_FACTOR = 2            # when hitting max_output_tokens, bump once by this factor


def _should_use_openai_responses(model_id: str) -> bool:
    normalized = model_id.strip().lower()
    return any(
        normalized.startswith(prefix) for prefix in ("gpt-4.1", "gpt-4o", "gpt-5", "o3", "o4")
    )


def _is_openai_gpt5_family(model_id: str) -> bool:
    return model_id.strip().lower().startswith("gpt-5")


def _extract_openai_responses_text(payload: Any) -> Optional[str]:
    """
    Extract visible text or JSON string from Responses API or Chat Completions payloads.

    Supports:
      - output_text: str | list[str]
      - output -> message -> content[] with parts:
          * {"type": "output_text" | "text" | "input_text", "text": str | {"value"/"content"/"text": str}}
          * {"type": "output_json" | "json", "json": dict}  # will be dumped to str
      - response.output[...] (some SDKs wrap as {"response": {...}})
      - message.content[...] (some SDKs expose "message" object)
      - top-level "content" list (rare)
      - chat.completions choices[0].message.content
    """
    if not isinstance(payload, dict):
        return None

    def _dump_json(obj: Any) -> Optional[str]:
        try:
            from json import dumps
            s = dumps(obj, ensure_ascii=False)
            return s if isinstance(s, str) and s.strip() else None
        except Exception:
            return None

    # 0) helper: scan a Responses-style "output" list
    def _scan_output_list(output_list: Any) -> Optional[str]:
        if not isinstance(output_list, list):
            return None
        for entry in output_list:
            if not isinstance(entry, dict):
                continue
            contents = entry.get("content") or []  # guard None
            if not isinstance(contents, list):
                continue
            for part in contents:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")

                # Structured JSON parts
                if part_type in ("output_json", "json"):
                    pj = part.get("json")
                    dumped = _dump_json(pj)
                    if dumped:
                        return dumped

                # Textual parts
                if part_type in ("output_text", "text", "input_text"):
                    text_field = part.get("text")
                    if isinstance(text_field, dict):
                        val = (
                            text_field.get("value")
                            or text_field.get("content")
                            or text_field.get("text")
                        )
                        if isinstance(val, str) and val.strip():
                            return val
                    if isinstance(text_field, str) and text_field.strip():
                        return text_field

                # Extra leniency
                if isinstance(part.get("text"), str) and part["text"].strip():
                    return part["text"].strip()

                # Nested tool_result -> content[] -> output_json
                if part_type == "tool_result":
                    nested = _scan_output_list(part.get("content"))
                    if nested:
                        return nested
        return None

    # 1) Fast path: aggregated text
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    if isinstance(output_text, list):
        for item in output_text:
            if isinstance(item, str) and item.strip():
                return item

    # 2) Some dumps wrap under "response"
    response_obj = payload.get("response")
    if isinstance(response_obj, dict):
        t = response_obj.get("output_text")
        if isinstance(t, str) and t.strip():
            return t
        nested = _scan_output_list(response_obj.get("output"))
        if nested:
            return nested

    # 3) Top-level Responses API shape
    top = _scan_output_list(payload.get("output"))
    if top:
        return top

    # 3.1) Some SDKs expose top-level "message" like Responses(Message)
    message_obj = payload.get("message")
    if isinstance(message_obj, dict):
        contents = message_obj.get("content")
        if isinstance(contents, list):
            for part in contents:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in ("output_text", "text", "input_text"):
                    tf = part.get("text")
                    if isinstance(tf, dict):
                        val = tf.get("value") or tf.get("content") or tf.get("text")
                        if isinstance(val, str) and val.strip():
                            return val
                    if isinstance(tf, str) and tf.strip():
                        return tf
                if part.get("type") in ("output_json", "json"):
                    dumped = _dump_json(part.get("json"))
                    if dumped:
                        return dumped

    # 3.2) Rare: top-level "content" directly
    top_content = payload.get("content")
    if isinstance(top_content, list):
        for part in top_content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("text", "output_text", "input_text"):
                tf = part.get("text")
                if isinstance(tf, dict):
                    val = tf.get("value") or tf.get("content") or tf.get("text")
                    if isinstance(val, str) and val.strip():
                        return val
                if isinstance(tf, str) and tf.strip():
                    return tf
            if part.get("type") in ("output_json", "json"):
                dumped = _dump_json(part.get("json"))
                if dumped:
                    return dumped

    # 4) Chat Completions compatibility
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") in ("text", "output_text"):
                            t = part.get("text")
                            if isinstance(t, dict):
                                val = t.get("value") or t.get("content") or t.get("text")
                                if isinstance(val, str) and val.strip():
                                    return val
                            if isinstance(t, str) and t.strip():
                                return t
                        # NEW: handle structured JSON in Chat content list
                        if isinstance(part, dict) and part.get("type") in ("output_json", "json"):
                            dumped = _dump_json(part.get("json"))
                            if dumped:
                                return dumped
            # Legacy: choice-level content list
            content_list = first_choice.get("content")
            if isinstance(content_list, list):
                for part in content_list:
                    if isinstance(part, dict) and part.get("type") in ("text", "output_text"):
                        t = part.get("text")
                        if isinstance(t, dict):
                            val = t.get("value") or t.get("content") or t.get("text")
                            if isinstance(val, str) and val.strip():
                                return val
                        if isinstance(t, str) and t.strip():
                            return t
                    # NEW: structured JSON in legacy content list
                    if isinstance(part, dict) and part.get("type") in ("output_json", "json"):
                        dumped = _dump_json(part.get("json"))
                        if dumped:
                            return dumped

    return None


def _openai_error_message(err: Exception) -> str:
    message = getattr(err, "message", None)
    if isinstance(message, str) and message.strip():
        return message.strip()
    body = getattr(err, "body", None)
    if isinstance(body, dict):
        error_field = body.get("error")
        if isinstance(error_field, dict):
            detail = error_field.get("message") or error_field.get("code")
            if isinstance(detail, str) and detail.strip():
                return detail.strip()
        elif isinstance(error_field, str) and error_field.strip():
            return error_field.strip()
    return str(err)


def _iterative_trim_call(create_fn, label: str, **kwargs):
    """
    Call an OpenAI SDK method and iteratively strip unsupported kwargs on TypeError/BadRequest.
    This keeps compatibility with older SDKs (e.g., 2.3.0) and server-side feature-gating.
    """
    UNSUPPORTED_CANDIDATES = [
        "response_format", "modalities", "reasoning",
        "instructions", "temperature", "max_completion_tokens", "max_tokens",
    ]
    attempts = 0
    while attempts < 6:
        try:
            return create_fn(**kwargs)
        except TypeError as e:
            msg = str(e)
            removed = False
            for k in list(kwargs.keys()):
                if any(k == bad and bad in msg for bad in UNSUPPORTED_CANDIDATES):
                    kwargs.pop(k, None)
                    removed = True
                    break
            if not removed:
                # conservative extra trim
                for bad in UNSUPPORTED_CANDIDATES:
                    if bad in kwargs:
                        kwargs.pop(bad, None)
                        removed = True
                        break
            if not removed:
                raise
            attempts += 1
        except BadRequestError as e:
            # server-side "Unsupported parameter" (e.g., temperature on gpt-5)
            msg = _openai_error_message(e)
            lowered = msg.lower()
            removed = False
            # Strip any obviously rejected parameters
            for bad in UNSUPPORTED_CANDIDATES:
                if bad in kwargs and bad in lowered:
                    kwargs.pop(bad, None)
                    removed = True
            if "temperature" in lowered and "temperature" in kwargs:
                kwargs.pop("temperature", None)
                removed = True
            if removed:
                attempts += 1
                continue
            raise
        except OpenAIError as e:
            msg = _openai_error_message(e)
            # Typical gpt-5 complaint: temperature unsupported
            if "temperature" in msg and "unsupported" in msg.lower():
                kwargs.pop("temperature", None)
                attempts += 1
                continue
            raise
    raise RuntimeError(f"{label} failed after trims")


def _call_openai_family(
    provider: str,
    model: str,
    api_key: str,
    system_content: str,
    user_content: str,
    score_min: float,
    score_max: float,
    timeout: float,
    *,
    base_url: Optional[str] = None,
    force_responses_api: bool = False,
    reasoning_effort: Optional[str] = None,  # "low" | "medium" | "high" | "none"/"off"/None
) -> RewardRubricRunResult:
    # --- Guard: SDK available ---
    if OpenAI is None or BadRequestError is None or OpenAIError is None:
        raise ProviderRequestError(
            provider,
            model,
            "OpenAI SDK is required. Install it via `pip install 'openai>=2.0.0'`.",
        )

    # --- Client / per-request options ---
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    req = client.with_options(timeout=timeout)  # per-request timeout (SDK v2)

    # --- Schema materials ---
    _ = reward_schema_definition()  # kept for parity/debug even if not used directly
    schema_payload = reward_json_schema()
    # Wrap bare JSON Schema into Responses v2 shape if necessary.
    if isinstance(schema_payload, dict) and "schema" not in schema_payload:
        schema_payload = {
            "name": "reward_score_schema",
            "strict": True,
            "schema": schema_payload,
        }

    # --- Temperature rules ---
    # gpt-5 family does NOT accept temperature; others should use 0 for determinism.
    is_gpt5 = _is_openai_gpt5_family(model)
    temperature_kwargs: Dict[str, Any] = {}
    if not is_gpt5:
        temperature_kwargs["temperature"] = 0

    # --- Build inputs ---
    # For Responses API: best practice -> system via `instructions`, user via `input` as `input_text`.
    input_user_only_input_text = [
        {"role": "user", "content": [{"type": "input_text", "text": user_content}]},
    ]

    # Chat Completions message shape (fallback)
    chat_messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    # --- Local helper: finalise ---
    def _finalise(raw_response: Any, text: Optional[str], parsed_obj: Any = None) -> RewardRubricRunResult:
        """
        - If parsed_obj is provided (dict/list), serialize and sanitize.
        - Else, try to sniff JSON objects in raw_response (output_json) and sanitize.
        - Else, fall back to text extraction / sanitize_json(text).
        """
        try:
            import json

            if parsed_obj is not None:
                parsed_str = json.dumps(parsed_obj, ensure_ascii=False)
                score, explanation = sanitize_json(parsed_str)
            else:
                # sniff output_json even if SDK doesn't provide output_parsed
                def _sniff_output_json(payload: Any) -> Optional[str]:
                    if not isinstance(payload, dict):
                        return None
                    for node in (payload, payload.get("response")):
                        if not isinstance(node, dict):
                            continue
                        out = node.get("output")
                        if isinstance(out, list):
                            for entry in out:
                                if not isinstance(entry, dict):
                                    continue
                                contents = entry.get("content")
                                if not isinstance(contents, list):
                                    continue
                                for part in contents:
                                    if isinstance(part, dict) and part.get("type") in ("output_json", "json"):
                                        pj = part.get("json")
                                        try:
                                            s = json.dumps(pj, ensure_ascii=False)
                                            if isinstance(s, str) and s.strip():
                                                return s
                                        except Exception:
                                            pass
                    return None

                json_str = _sniff_output_json(raw_response)
                if json_str:
                    score, explanation = sanitize_json(json_str)
                else:
                    if not text:
                        text = _extract_openai_responses_text(raw_response)
                    if not text:
                        # Surface incomplete status for better error messages
                        status = None
                        incomplete = None
                        if isinstance(raw_response, dict):
                            node = raw_response.get("response") if isinstance(raw_response.get("response"), dict) else raw_response
                            status = node.get("status")
                            incomplete = node.get("incomplete_details")
                        if status and status != "completed":
                            reason = (incomplete or {}).get("reason")
                            raise ProviderRequestError(provider, model, f"Response incomplete: status={status}, reason={reason or 'unknown'}")
                        raise ProviderRequestError(provider, model, "Model response did not include any content.")
                    score, explanation = sanitize_json(text)

        except ValueError as err:
            raise ProviderRequestError(provider, model, str(err)) from err

        bounded = max(score_min, min(score_max, score))
        return {"score": bounded, "explanation": explanation, "raw": raw_response}

    # --- Helper: try Responses with a given response_format mode and adaptive bump ---
    def _try_responses(mode: str) -> Optional[RewardRubricRunResult]:
        """
        mode: "json_schema" | "json_object" | "none"
        Returns a RewardRubricRunResult or None to indicate fallback is needed.
        """
        kwargs_rf: Dict[str, Any] = {}
        if mode == "json_schema":
            kwargs_rf["response_format"] = {"type": "json_schema", "json_schema": schema_payload}
        elif mode == "json_object":
            kwargs_rf["response_format"] = {"type": "json_object"}

        responses_base: Dict[str, Any] = {
            "model": model,
            "instructions": system_content,        # system here
            "input": input_user_only_input_text,   # user only; input_text
            "max_output_tokens": SAFE_DEFAULT_MAX_OUTPUT,
            "modalities": ["text"],                # be explicit; will be trimmed if unsupported
        }
        # gpt-5 must not receive temperature
        if not is_gpt5:
            responses_base.update(temperature_kwargs)

        # Optional reasoning effort knob, if provided
        effort = (reasoning_effort or "").strip().lower() if reasoning_effort else None
        if effort in ("low", "medium", "high"):
            responses_base["reasoning"] = {"effort": effort}
        # "none"/"off"/None => do not send "reasoning" kw

        try:
            # First attempt
            response = _iterative_trim_call(req.responses.create, "responses", **{**responses_base, **kwargs_rf})
            raw = dump_model(response)
            # If incomplete due to token ceiling, bump once and retry
            try:
                node = raw.get("response", raw) if isinstance(raw, dict) else {}
                if isinstance(node, dict) and node.get("status") == "incomplete":
                    inc = node.get("incomplete_details") or {}
                    if inc.get("reason") == "max_output_tokens":
                        bumped = {**responses_base, **kwargs_rf}
                        bumped["max_output_tokens"] = min(max(SAFE_DEFAULT_MAX_OUTPUT * SAFE_BUMP_FACTOR, 2048), SAFE_MAX_OUTPUT_CAP)
                        response_b = _iterative_trim_call(req.responses.create, "responses-bumped", **bumped)
                        raw_b = dump_model(response_b)

                        parsed_b = getattr(response_b, "output_parsed", None)
                        if parsed_b is not None:
                            return _finalise(raw_b, None, parsed_obj=parsed_b)

                        text_b = getattr(response_b, "output_text", None) or _extract_openai_responses_text(raw_b)
                        if text_b:
                            return _finalise(raw_b, text_b)
                        # else: fall through to chat fallback
                        return None
            except Exception:
                # if any inspection error, just proceed to parse normally
                pass

            # Normal parse path
            parsed = getattr(response, "output_parsed", None)
            if parsed is not None:
                return _finalise(raw, None, parsed_obj=parsed)

            text = getattr(response, "output_text", None) or _extract_openai_responses_text(raw)
            if text:
                return _finalise(raw, text)

            # No content -> let caller try next mode or chat
            return None

        except BadRequestError as err:
            msg = _openai_error_message(err).lower()
            # If backend complains about json_schema, try json_object / none next
            if "response_format" in msg and "json_schema" in msg and mode == "json_schema":
                return None  # caller will try json_object
            if "response_format" in msg and mode in ("json_schema", "json_object"):
                return None  # caller will try 'none'
            # Other errors -> raise
            raise ProviderRequestError(provider, model, f"Model request failed. {msg}") from err

        except OpenAIError as err:
            msg = _openai_error_message(err)
            raise ProviderRequestError(provider, model, f"Model request failed. {msg}") from err

    # --- Decide primary path ---
    use_responses_api = force_responses_api or _should_use_openai_responses(model)

    # --------- RESPONSES API PATH ---------
    if use_responses_api:
        # Try json_schema -> json_object -> none
        for mode in ("json_schema", "json_object", "none"):
            result = _try_responses(mode)
            if result is not None:
                return result
        # fall through to Chat if Responses yielded nothing

    # --------- CHAT COMPLETIONS PATH ---------
    try:
        fam = model.strip().lower().split(":")[0]
        tokens_kw: Dict[str, Any] = {}
        if any(fam.startswith(p) for p in FAMILIES_USE_MAX_COMPLETION):
            tokens_kw["max_completion_tokens"] = SAFE_DEFAULT_MAX_OUTPUT
        else:
            tokens_kw["max_tokens"] = SAFE_DEFAULT_MAX_OUTPUT

        # build response_format for Chat
        chat_rf: Dict[str, Any] = {}
        # We attempt to send json_schema; if SDK rejects, _iterative_trim_call will remove.
        chat_rf = {"response_format": {"type": "json_schema", "json_schema": schema_payload}}

        chat_base: Dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            **tokens_kw,
            **chat_rf,
        }
        if not is_gpt5:
            chat_base.update(temperature_kwargs)  # temperature=0 for non-gpt-5

        completion = _iterative_trim_call(req.chat.completions.create, "chat.completions", **chat_base)
        raw = dump_model(completion)
        text = _extract_openai_responses_text(raw)
        return _finalise(raw, text)

    except OpenAIError as err:
        msg = _openai_error_message(err)
        raise ProviderRequestError(provider, model, f"Model request failed. {msg}") from err


class OpenAIProvider(RubricProvider):
    name = "openai"

    def default_timeout(self, model: str) -> float:
        return DEFAULT_REQUEST_TIMEOUT_SECONDS

    def run(self, request: ProviderRequest) -> RewardRubricRunResult:
        # Try to fetch reasoning effort hint from request if present
        effort_hint: Optional[str] = getattr(request, "reasoning_effort", None)
        opts = getattr(request, "options", None)
        if effort_hint is None and isinstance(opts, dict):
            effort_hint = opts.get("reasoning_effort") or opts.get("effort")

        return _call_openai_family(
            provider=self.name,
            model=request.model,
            api_key=request.api_key,
            system_content=request.system_content,
            user_content=request.user_content,
            score_min=request.score_min,
            score_max=request.score_max,
            timeout=request.timeout,
            reasoning_effort=effort_hint,
        )


class XAIProvider(OpenAIProvider):
    name = "xai"

    def default_timeout(self, model: str) -> float:
        normalized = model.strip().lower()
        if normalized.startswith("grok-4"):
            return 60.0
        return 45.0

    def run(self, request: ProviderRequest) -> RewardRubricRunResult:
        # Try to fetch reasoning effort hint from request if present
        effort_hint: Optional[str] = getattr(request, "reasoning_effort", None)
        opts = getattr(request, "options", None)
        if effort_hint is None and isinstance(opts, dict):
            effort_hint = opts.get("reasoning_effort") or opts.get("effort")

        return _call_openai_family(
            provider=self.name,
            model=request.model,
            api_key=request.api_key,
            system_content=request.system_content,
            user_content=request.user_content,
            score_min=request.score_min,
            score_max=request.score_max,
            timeout=request.timeout,
            base_url="https://api.x.ai/v1",
            force_responses_api=True,
            reasoning_effort=effort_hint,
        )


__all__ = [
    "OpenAIProvider",
    "XAIProvider",
]
