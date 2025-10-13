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

MESSAGES = [
    {
        "type": "message",
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": (
                    "My AirPure X2 purifier stopped working. The power light keeps blinking "
                    "and I really need it running before guests arrive tomorrow. Can you help?"
                ),
            }
        ],
    },
    {
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": (
                    "Absolutely—I can help troubleshoot this. To confirm your warranty status, "
                    "could you share the order number or the phone number used for purchase? "
                    "Also, is the blinking light red or green, and have you tried any steps like "
                    "power cycling or checking the filter yet?"
                ),
            }
        ],
    },
    {
        "type": "message",
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": (
                    "The order number is 81234-AX2. The power light is a slow red blink. I replaced "
                    "the filter yesterday but haven't done anything else."
                ),
            }
        ],
    },
    {
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": (
                    "Thanks, that order number shows you are still under warranty. A red slow blink "
                    "usually means the fan safety cut-off engaged. Please unplug the purifier, remove "
                    "the base panel, and check if any packing foam or debris is touching the fan blades. "
                    "If the fan moves freely, plug it back in after five minutes to clear the sensor. "
                    "If it still blinks, I can schedule a technician tomorrow morning—does that fit your "
                    "timeline before your guests arrive?"
                ),
            }
        ],
    },
]

GROUND_TRUTH = (
    "The assistant should confirm warranty details, gather diagnostics about the blinking light, "
    "suggest safe troubleshooting steps, and offer a timely service appointment."
)


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
            "model": "gpt-5-nano-2025-08-07",
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
