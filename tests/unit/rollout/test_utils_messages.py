"""Tests for osmosis_ai.rollout.utils.messages."""

from osmosis_ai.rollout.utils.messages import map_initial_messages_to_content_blocks


class TestMapInitialMessagesToContentBlocks:
    def test_single_message(self):
        msgs = [{"role": "user", "content": "Hello"}]
        result = map_initial_messages_to_content_blocks(msgs)
        assert result == [{"role": "user", "content": [{"text": "Hello"}]}]

    def test_multi_turn(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        result = map_initial_messages_to_content_blocks(msgs)
        assert len(result) == 2
        assert result[0] == {
            "role": "system",
            "content": [{"text": "You are helpful."}],
        }
        assert result[1] == {"role": "user", "content": [{"text": "Hi"}]}

    def test_empty_list(self):
        assert map_initial_messages_to_content_blocks([]) == []
