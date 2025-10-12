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
)

RUBRIC = "Assistant must mention the verified capital city and stay on topic."
SCORE_MIN = 0.0
SCORE_MAX = 1.0

MESSAGES = [
    {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "What is the capital of France?"}],
    },
    {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "Paris is the capital city of France."}],
    },
]

GROUND_TRUTH = "Paris"


def _run(provider_name: str, model_info: dict) -> None:
    try:
        result = evaluate_rubric(
            rubric=RUBRIC,
            messages=MESSAGES,
            ground_truth=GROUND_TRUTH,
            model_info=model_info,
            score_min=SCORE_MIN,
            score_max=SCORE_MAX,
            return_details=True,
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

    print(f"{provider_name} score: {result['score']:.2f} (range {SCORE_MIN}-{SCORE_MAX})")
    print(f"{provider_name} explanation: {result['explanation']}")


def run_openai_example() -> None:
    _run(
        "OpenAI",
        {
            "provider": "openai",
            "model": "gpt-5",
        },
    )


def run_anthropic_example() -> None:
    _run(
        "Anthropic",
        {
            "provider": "anthropic",
            "model": "claude-3-7-sonnet-20250219",
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
