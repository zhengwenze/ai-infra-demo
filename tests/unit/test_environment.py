from pathlib import Path

from inferscope.environment import capture_environment, capture_git_state


def test_capture_environment_uses_whitelisted_sections(tmp_path: Path) -> None:
    captured = capture_environment(tmp_path)

    assert set(captured) == {"schema_version", "platform", "packages", "git", "nvidia"}
    assert captured["schema_version"] == "1.0"
    assert captured["platform"]["python_version"]
    assert "environment" not in captured


def test_capture_git_state_tolerates_non_git_directory(tmp_path: Path) -> None:
    state = capture_git_state(tmp_path)

    assert state == {"available": False, "commit": None, "branch": None, "dirty": None}
