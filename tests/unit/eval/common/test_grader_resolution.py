import sys
import types

import pytest

from osmosis_ai.eval.common.cli import _resolve_grader


@pytest.fixture
def fake_module_with_grader():
    """Create a fake module with a Grader subclass and GraderConfig."""
    from osmosis_ai.rollout_v2.context import GraderContext
    from osmosis_ai.rollout_v2.grader import Grader
    from osmosis_ai.rollout_v2.types import GraderConfig

    class FakeGrader(Grader):
        async def grade(self, ctx: GraderContext):
            pass

    fake_config = GraderConfig(name="test_grader")

    mod = types.ModuleType("fake_grader_mod")
    mod.FakeGrader = FakeGrader
    mod.grader_config = fake_config
    sys.modules["fake_grader_mod"] = mod
    yield mod
    del sys.modules["fake_grader_mod"]


@pytest.fixture
def fake_module_without_grader():
    """Create a fake module with no Grader."""
    mod = types.ModuleType("fake_no_grader_mod")
    mod.SomeClass = type("SomeClass", (), {})
    sys.modules["fake_no_grader_mod"] = mod
    yield mod
    del sys.modules["fake_no_grader_mod"]


def test_resolve_grader_auto_discover(fake_module_with_grader):
    from osmosis_ai.rollout_v2.grader import Grader
    from osmosis_ai.rollout_v2.types import GraderConfig

    cls, config = _resolve_grader("fake_grader_mod")
    assert cls is not None
    assert issubclass(cls, Grader)
    assert isinstance(config, GraderConfig)


def test_resolve_grader_not_found(fake_module_without_grader):
    cls, config = _resolve_grader("fake_no_grader_mod")
    assert cls is None
    assert config is None


def test_resolve_grader_explicit_path(fake_module_with_grader):
    cls, config = _resolve_grader(
        "fake_grader_mod",
        explicit_grader="fake_grader_mod:FakeGrader",
        explicit_config="fake_grader_mod:grader_config",
    )
    assert cls is not None
    assert cls.__name__ == "FakeGrader"
    assert config is not None
