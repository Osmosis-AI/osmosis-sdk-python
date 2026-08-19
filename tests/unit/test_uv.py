"""Locating the uv executable, shared by bundle packaging and local eval."""

import pytest

import osmosis_ai._uv as uv_module


def test_uv_executable_uses_active_scripts_scheme(tmp_path, monkeypatch):
    scripts_dir = tmp_path / "Scripts"
    scripts_dir.mkdir()
    uv = scripts_dir / "uv.exe"
    uv.touch()
    uv.chmod(0o755)

    def get_path(name):
        assert name == "scripts"
        return str(scripts_dir)

    monkeypatch.setattr(uv_module.sysconfig, "get_path", get_path)
    monkeypatch.setattr(uv_module.sys, "executable", str(tmp_path / "python.exe"))
    monkeypatch.setattr(uv_module.shutil, "which", lambda _name: None)

    assert uv_module._uv_executable() == str(uv)


def test_uv_executable_reports_missing_builder(tmp_path, monkeypatch):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    monkeypatch.setattr(uv_module.sysconfig, "get_path", lambda name: str(scripts_dir))
    monkeypatch.setattr(uv_module.sys, "executable", str(tmp_path / "python"))
    monkeypatch.setattr(uv_module.shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match=r"install osmosis-ai\[harbor\]"):
        uv_module._uv_executable()


def test_uv_executable_skips_non_executable_candidate(tmp_path, monkeypatch):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    candidate = scripts_dir / "uv"
    candidate.touch()
    fallback = tmp_path / "path" / "uv"
    monkeypatch.setattr(uv_module.sysconfig, "get_path", lambda name: str(scripts_dir))
    monkeypatch.setattr(uv_module.sys, "executable", str(tmp_path / "python"))
    monkeypatch.setattr(uv_module.os, "access", lambda path, mode: path != candidate)
    monkeypatch.setattr(uv_module.shutil, "which", lambda _name: str(fallback))

    assert uv_module._uv_executable() == str(fallback)
