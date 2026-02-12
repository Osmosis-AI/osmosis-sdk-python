"""
Evaluate the same rubric against conversations using different judge providers.

Set the following environment variables before running the examples:

    export OPENAI_API_KEY="..."
    export ANTHROPIC_API_KEY="..."
    export GEMINI_API_KEY="..."
    export XAI_API_KEY="..."
    export OPENROUTER_API_KEY="..."
    export CEREBRAS_API_KEY="..."

Uncomment the desired provider in the `__main__` section to trigger a request.

Each helper call uses LiteLLM's unified interface with structured JSON outputs
enforced. Any provider supported by LiteLLM can be used by passing its name
in `model_info`.
"""

from __future__ import annotations

from osmosis_ai import (
    MissingAPIKeyError,
    ModelNotFoundError,
    ProviderRequestError,
    evaluate_rubric,
    osmosis_rubric,
)
from osmosis_ai.rubric_eval import DEFAULT_API_KEY_ENV

# Rubric for the example
RUBRIC = (
    "Evaluate the assistant's ability to handle a smart appliance support case. "
    "Award higher scores when the assistant: (1) confirms purchase information to "
    "verify warranty status, (2) gathers relevant troubleshooting details (error "
    "lights, recent maintenance, prior attempts), (3) offers safe, actionable next "
    "steps the user can try immediately, and (4) sets expectations for follow-up "
    "service or scheduling while acknowledging the user's urgency. Deduct points if "
    "the assistant ignores the user's constraints, invents policies, or makes unsafe "
    "suggestions. Base the score solely on the conversation."
)
SCORE_MIN = 0.0
SCORE_MAX = 1.0

# Data for the example
SOLUTION_STR = (
    "Thanks, that order number shows you are still under warranty. A red slow blink usually means "
    "the fan safety cut-off engaged. Please unplug the purifier, remove the base panel, and check if "
    "any packing foam or debris is touching the fan blades. If the fan moves freely, plug it back in "
    "after five minutes to clear the sensor. If it still blinks, I can schedule a technician tomorrow "
    "morning—does that fit your timeline before your guests arrive?"
)

GROUND_TRUTH = (
    "The assistant should confirm warranty details, gather diagnostics about the blinking light, "
    "suggest safe troubleshooting steps, and offer a timely service appointment."
)

PROFILE_CATALOG = {
    "openai": {
        "provider": "openai",
        "model": "gpt-5-nano-2025-08-07",
        "api_key_env": DEFAULT_API_KEY_ENV["openai"],
    },
    "anthropic": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5-20250929",
        "api_key_env": DEFAULT_API_KEY_ENV["anthropic"],
    },
    "gemini": {
        "provider": "gemini",
        "model": "gemini-3-flash-preview",
        "api_key_env": DEFAULT_API_KEY_ENV["gemini"],
    },
    "xai": {
        "provider": "xai",
        "model": "grok-4-fast-non-reasoning",
        "api_key_env": DEFAULT_API_KEY_ENV["xai"],
    },
    "openrouter": {
        "provider": "openrouter",
        "model": "openai/gpt-oss-safeguard-20b",
        "api_key_env": DEFAULT_API_KEY_ENV["openrouter"],
    },
    "cerebras": {
        "provider": "cerebras",
        "model": "qwen-3-235b-a22b-instruct-2507",
        "api_key_env": DEFAULT_API_KEY_ENV["cerebras"],
    },
}


@osmosis_rubric
def score_with_hosted_model(
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
) -> float:
    """
    Delegate rubric scoring to a hosted model while keeping @osmosis_rubric validation.

    Provide provider-specific knobs inside `extra_info["metadata"]`. Toggle `extra_info["capture_details"]`
    when you want the provider response returned.
    """
    metadata = extra_info.get("metadata") if isinstance(extra_info, dict) and isinstance(extra_info.get("metadata"), dict) else None
    capture_details = bool(extra_info.get("capture_details")) if isinstance(extra_info, dict) else False
    prompt_metadata = metadata

    provider_config = _resolve_provider_profile(metadata.get("provider_profile") if metadata else None)

    model_info = dict(provider_config["model_info"])

    rubric = provider_config["rubric"]
    score_min = provider_config["score_min"]
    score_max = provider_config["score_max"]

    result = evaluate_rubric(
        rubric=rubric,
        solution_str=solution_str,
        model_info=model_info,
        ground_truth=ground_truth,
        metadata=prompt_metadata,
        score_min=score_min,
        score_max=score_max,
        return_details=capture_details,
    )

    if capture_details:
        # Surface the provider response via the metadata channel so callers can inspect it.
        if metadata is not None:
            metadata["result_details"] = result
        return float(result["score"])

    return float(result)


def _normalize_profile_name(profile_name: str | None) -> str:
    if isinstance(profile_name, str):
        normalized = profile_name.strip().lower()
        if normalized:
            return normalized
    return "openai"


def _resolve_provider_profile(profile_name: str | None) -> dict:
    profile_key = _normalize_profile_name(profile_name)

    profile = PROFILE_CATALOG.get(profile_key)
    if profile is None:
        options = ", ".join(sorted(PROFILE_CATALOG))
        raise ValueError(f"Unknown provider_profile '{profile_name}'. Supported profiles: {options}")

    return {
        "model_info": profile,
        "rubric": RUBRIC,
        "score_min": SCORE_MIN,
        "score_max": SCORE_MAX,
    }


def _run(provider_name: str, provider_profile: str) -> None:
    try:
        context: dict = {
            "capture_details": True,
            "metadata": {
                "provider_profile": provider_profile,
                "scenario_label": provider_name,
            }
        }
        score = score_with_hosted_model(
            solution_str=SOLUTION_STR,
            ground_truth=GROUND_TRUTH,
            extra_info=context,
        )
    except MissingAPIKeyError as exc:
        print(f"{provider_name} skipped: {exc}")
        return
    except ModelNotFoundError as exc:
        print(f"{provider_name} skipped: {exc.detail}")
        return
    except ProviderRequestError as exc:
        print(f"{provider_name} failed: {exc.detail}")
        return

    metadata = context.get("metadata", {})
    details = metadata.get("result_details")
    explanation = ""
    if isinstance(details, dict):
        explanation = details.get("explanation", "")
    print(f"{provider_name} score: {score:.2f} (range {SCORE_MIN}-{SCORE_MAX})")
    print(f"{provider_name} explanation: {explanation}")


def run_openai_example() -> None:
    _run(
        "OpenAI",
        "openai",
    )


def run_anthropic_example() -> None:
    _run(
        "Anthropic",
        "anthropic",
    )


def run_gemini_example() -> None:
    _run(
        "Gemini",
        "gemini",
    )


def run_xai_example() -> None:
    _run(
        "xAI",
        "xai",
    )


def run_openrouter_example() -> None:
    _run(
        "OpenRouter",
        "openrouter",
    )


def run_cerebras_example() -> None:
    _run(
        "Cerebras",
        "cerebras",
    )


if __name__ == "__main__":
    # Uncomment the provider calls you want to exercise:
    # run_openai_example()
    # run_anthropic_example()
    # run_gemini_example()
    # run_xai_example()
    run_openrouter_example()
    run_cerebras_example()
    pass
