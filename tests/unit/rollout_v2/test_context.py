from osmosis_ai.rollout_v2.context import RolloutContext, get_rollout_context


def test_nested_context_restores_outer():
    """After exiting inner context, outer context should be restored."""
    outer = RolloutContext(chat_completions_url="http://outer", rollout_id="outer")
    inner = RolloutContext(chat_completions_url="http://inner", rollout_id="inner")

    with outer:
        assert get_rollout_context() is outer
        with inner:
            assert get_rollout_context() is inner
        assert get_rollout_context() is outer

    assert get_rollout_context() is None


def test_single_context_sets_and_clears():
    """Single context manager still works correctly."""
    ctx = RolloutContext(chat_completions_url="http://test", rollout_id="test")
    assert get_rollout_context() is None
    with ctx:
        assert get_rollout_context() is ctx
    assert get_rollout_context() is None
