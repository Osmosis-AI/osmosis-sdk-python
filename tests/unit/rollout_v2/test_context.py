from osmosis_ai.rollout_v2.context import (
    RolloutContext,
    get_rollout_context,
    rollout_contextvar,
)


def test_single_context_sets_and_clears():
    """Single context manager still works correctly."""
    ctx = RolloutContext(chat_completions_url="http://test", rollout_id="test")
    assert get_rollout_context() is None
    with ctx:
        assert get_rollout_context() is ctx
    assert get_rollout_context() is None


def test_contextvar_reset_restores_outer():
    """InProcessDriver pattern: set/reset on contextvar preserves outer value."""
    outer = RolloutContext(chat_completions_url="http://outer", rollout_id="outer")
    inner = RolloutContext(chat_completions_url="http://inner", rollout_id="inner")

    with outer:
        assert get_rollout_context() is outer
        # InProcessDriver uses contextvar.set/reset directly (not with)
        token = rollout_contextvar.set(inner)
        try:
            assert get_rollout_context() is inner
        finally:
            rollout_contextvar.reset(token)
        assert get_rollout_context() is outer

    assert get_rollout_context() is None
