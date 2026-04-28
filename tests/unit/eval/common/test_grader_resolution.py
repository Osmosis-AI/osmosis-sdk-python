import sys
import types

import pytest

from osmosis_ai.errors import CLIError
from osmosis_ai.eval.common.cli import _resolve_grader


@pytest.fixture
def fake_module_with_grader():
    """Create a fake module with a Grader subclass and GraderConfig."""
    from osmosis_ai.rollout.context import GraderContext
    from osmosis_ai.rollout.grader import Grader
    from osmosis_ai.rollout.types import GraderConfig

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
    from osmosis_ai.rollout.grader import Grader
    from osmosis_ai.rollout.types import GraderConfig

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


def test_resolve_grader_explicit_not_a_class(fake_module_with_grader):
    """Explicit grader pointing to a function should raise CLIError."""
    fake_module_with_grader.some_func = lambda: None

    with pytest.raises(CLIError, match="concrete Grader subclass"):
        _resolve_grader(
            "fake_grader_mod",
            explicit_grader="fake_grader_mod:some_func",
        )


def test_resolve_grader_explicit_wrong_class(fake_module_with_grader):
    """Explicit grader pointing to a non-Grader class should raise CLIError."""
    fake_module_with_grader.NotAGrader = type("NotAGrader", (), {})

    with pytest.raises(CLIError, match="concrete Grader subclass"):
        _resolve_grader(
            "fake_grader_mod",
            explicit_grader="fake_grader_mod:NotAGrader",
        )


def test_resolve_grader_explicit_abstract_grader(fake_module_with_grader):
    """Explicit grader pointing to the abstract Grader base should raise CLIError."""
    from osmosis_ai.rollout.grader import Grader

    fake_module_with_grader.BaseGrader = Grader

    with pytest.raises(CLIError, match="concrete Grader subclass"):
        _resolve_grader(
            "fake_grader_mod",
            explicit_grader="fake_grader_mod:BaseGrader",
        )


def test_resolve_grader_explicit_bad_config(fake_module_with_grader):
    """Explicit grader config pointing to a non-GraderConfig should raise CLIError."""
    fake_module_with_grader.bad_config = {"not": "a config"}

    with pytest.raises(CLIError, match="GraderConfig instance"):
        _resolve_grader(
            "fake_grader_mod",
            explicit_grader="fake_grader_mod:FakeGrader",
            explicit_config="fake_grader_mod:bad_config",
        )
