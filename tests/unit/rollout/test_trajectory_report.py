"""Tests for osmosis_ai.rollout.trajectory.report."""

import httpx

from osmosis_ai.rollout.trajectory.report import report_from_response


def test_parses_trajectory_object_from_ack_body() -> None:
    response = httpx.Response(
        200,
        json={
            "ok": True,
            "trajectory": {
                "model_name": "openai/gpt-5-mini",
                "samples": {
                    "s1": {
                        "llm_calls": [
                            {"prompt_tokens": 10, "completion_tokens": 5},
                            {"prompt_tokens": 20, "completion_tokens": 7},
                        ]
                    }
                },
            },
        },
    )

    report = report_from_response(response)

    assert report is not None
    assert report.model_name == "openai/gpt-5-mini"
    assert len(report.samples["s1"].llm_calls) == 2
    assert report.samples["s1"].llm_calls[1].completion_tokens == 7


def test_parses_token_ids_for_training_grade_reports() -> None:
    response = httpx.Response(
        200,
        json={
            "trajectory": {
                "samples": {
                    "s1": {
                        "llm_calls": [
                            {
                                "prompt_token_ids": [101, 102],
                                "completion_token_ids": [103],
                                "logprobs": [-0.5],
                            }
                        ]
                    }
                }
            }
        },
    )

    report = report_from_response(response)

    assert report is not None
    call = report.samples["s1"].llm_calls[0]
    assert call.prompt_token_ids == [101, 102]
    assert call.completion_token_ids == [103]
    assert call.logprobs == [-0.5]


def test_body_without_trajectory_yields_none() -> None:
    assert report_from_response(httpx.Response(200, json={"ok": True})) is None


def test_non_json_body_yields_none() -> None:
    assert report_from_response(httpx.Response(200, text="ok")) is None
    assert report_from_response(httpx.Response(200, json=[1, 2])) is None


def test_malformed_trajectory_yields_none() -> None:
    response = httpx.Response(
        200, json={"trajectory": {"samples": {"s1": {"llm_calls": "oops"}}}}
    )

    assert report_from_response(response) is None


def test_unknown_keys_are_ignored() -> None:
    # Forward compatibility: fields added to the protocol later must not
    # break older SDKs.
    response = httpx.Response(
        200,
        json={
            "trajectory": {
                "model_name": "m",
                "future_field": {"x": 1},
                "samples": {"s1": {"llm_calls": [], "another_new_field": 2}},
            }
        },
    )

    report = report_from_response(response)

    assert report is not None
    assert report.model_name == "m"
