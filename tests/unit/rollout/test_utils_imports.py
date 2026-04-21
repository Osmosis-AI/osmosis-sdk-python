"""Tests for osmosis_ai.rollout.utils.imports."""

import pytest

from osmosis_ai.rollout.utils.imports import resolve_object, to_import_path

# ---------------------------------------------------------------------------
# resolve_object
# ---------------------------------------------------------------------------


class TestResolveObject:
    def test_passthrough_non_string(self):
        obj = {"key": "value"}
        assert resolve_object(obj) is obj

    def test_resolves_valid_import_path(self):
        cls = resolve_object("osmosis_ai.rollout.utils.imports:resolve_object")
        assert cls is resolve_object

    def test_raises_on_missing_colon(self):
        with pytest.raises(ValueError, match="expected 'module:attr' format"):
            resolve_object("osmosis_ai.rollout.utils.imports")

    def test_raises_on_nonexistent_attr(self):
        with pytest.raises(ImportError, match="does not export 'NoSuchThing'"):
            resolve_object("osmosis_ai.rollout.utils.imports:NoSuchThing")

    def test_raises_on_nonexistent_module(self):
        with pytest.raises(ModuleNotFoundError):
            resolve_object("nonexistent.module:Foo")


# ---------------------------------------------------------------------------
# to_import_path
# ---------------------------------------------------------------------------


class TestToImportPath:
    def test_class_roundtrip(self):
        from osmosis_ai.rollout.utils.concurrency import ConcurrencyLimiter

        path = to_import_path(ConcurrencyLimiter)
        assert path == "osmosis_ai.rollout.utils.concurrency:ConcurrencyLimiter"
        assert resolve_object(path) is ConcurrencyLimiter

    def test_raises_for_unresolvable_object(self):
        # A locally created lambda cannot be found in any module's namespace.
        with pytest.raises(ValueError, match="Cannot resolve import path"):
            to_import_path(lambda: None)
