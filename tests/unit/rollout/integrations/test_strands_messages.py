"""Tests for Strands initial-message content-block conversion."""

from osmosis_ai.rollout.integrations.agents.strands import _content_block_messages


class TestContentBlockMessages:
    def test_multi_turn(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        result = _content_block_messages(msgs)
        assert len(result) == 2
        assert result[0] == {
            "role": "system",
            "content": [{"text": "You are helpful."}],
        }
        assert result[1] == {"role": "user", "content": [{"text": "Hi"}]}
