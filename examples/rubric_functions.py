"""
Evaluate the same rubric against conversations using different judge providers.

Set the following environment variables before running the examples:

    export OPENAI_API_KEY="..."
    export ANTHROPIC_API_KEY="..."
    export GOOGLE_API_KEY="..."
    export XAI_API_KEY="..."

Uncomment the desired provider in the `__main__` section to trigger a request.

Each helper call uses the provider's official Python SDK with structured JSON outputs
enforced. Providers are pluggable; see `osmosis_ai/providers/README.md` for instructions
on registering your own integration and then pass its name in `model_info`.
"""

from __future__ import annotations

from osmosis_ai import (
    MissingAPIKeyError,
    ModelNotFoundError,
    ProviderRequestError,
    evaluate_rubric,
    osmosis_rubric,
)

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


@osmosis_rubric
def score_with_hosted_model(
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
) -> float:
    """
    Delegate rubric scoring to a hosted model while keeping @osmosis_rubric validation.
    """
    capture_details = bool(extra_info.get("capture_details"))
    prompt_extra = extra_info.get("prompt_extra_info")
    model_info = extra_info.get("model_info")
    if not isinstance(model_info, dict):
        raise TypeError("extra_info must include a 'model_info' mapping")

    rubric = extra_info.get("rubric", RUBRIC)
    score_min = extra_info.get("score_min", SCORE_MIN)
    score_max = extra_info.get("score_max", SCORE_MAX)

    result = evaluate_rubric(
        rubric=rubric,
        solution_str=solution_str,
        model_info=model_info,
        ground_truth=ground_truth,
        extra_info=prompt_extra,
        score_min=score_min,
        score_max=score_max,
        return_details=capture_details,
    )

    if capture_details:
        # Treat extra_info as an input/output channel to surface detailed results.
        if extra_info is not None:
            extra_info["result_details"] = result
        return float(result["score"])

    return float(result)


def _run(provider_name: str, model_info: dict) -> None:
    try:
        provider_id = model_info["provider"]
        model_name = model_info["model"]
        api_key_env = model_info.get("api_key_env")
        if not isinstance(api_key_env, str) or not api_key_env.strip():
            provider_env_defaults = {"gemini": "GOOGLE_API_KEY"}
            provider_key = provider_id.strip().lower()
            api_key_env = provider_env_defaults.get(provider_key)
            if not api_key_env:
                api_key_env = f"{provider_key.upper()}_API_KEY"
        context: dict = {
            "provider": provider_id,
            "model": model_name,
            "api_key_env": api_key_env,
            "model_info": {**model_info, "api_key_env": api_key_env},
            "rubric": RUBRIC,
            "score_min": SCORE_MIN,
            "score_max": SCORE_MAX,
            "capture_details": True,
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

    details = context.get("result_details")
    explanation = ""
    if isinstance(details, dict):
        explanation = details.get("explanation", "")
    print(f"{provider_name} score: {score:.2f} (range {SCORE_MIN}-{SCORE_MAX})")
    print(f"{provider_name} explanation: {explanation}")


def run_openai_example() -> None:
    _run(
        "OpenAI",
        {
            "provider": "openai",
            "model": "gpt-5-nano-2025-08-07",
        },
    )


def run_anthropic_example() -> None:
    _run(
        "Anthropic",
        {
            "provider": "anthropic",
            "model": "claude-sonnet-4-5-20250929",
        },
    )


def run_gemini_example() -> None:
    _run(
        "Gemini",
        {
            "provider": "gemini",
            "model": "gemini-2.5-flash",
        },
    )


def run_xai_example() -> None:
    _run(
        "xAI",
        {
            "provider": "xai",
            "model": "grok-4-fast-non-reasoning",
        },
    )


if __name__ == "__main__":
    # Uncomment the provider calls you want to exercise:
    run_openai_example()
    run_anthropic_example()
    run_gemini_example()
    run_xai_example()
    pass
