def make_task(path, separate=False):
    (path / "environment").mkdir(parents=True)
    (path / "environment/Dockerfile").write_text("FROM ubuntu:24.04\nCOPY data /data\n")
    (path / "environment/data").write_text("content")
    (path / "instruction.md").write_text("Do something")
    (path / "tests").mkdir()
    (path / "tests/test.sh").write_text("echo 1 > /logs/verifier/reward.txt\n")
    (path / "task.toml").write_text(
        "[environment]\ncpus = 3\n"
        + ('[verifier]\nenvironment_mode = "separate"\n' if separate else "")
    )
    return path
