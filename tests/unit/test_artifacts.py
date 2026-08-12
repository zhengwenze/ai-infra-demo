import json
from pathlib import Path

import pytest

from inferscope.artifacts import ArtifactStore
from inferscope.errors import ConfigurationError


def test_artifact_store_creates_immutable_run_directory(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, "run-001")
    run_dir = store.create()

    assert run_dir == tmp_path / "raw" / "run-001"
    assert (run_dir / "logs").is_dir()
    with pytest.raises(FileExistsError):
        store.create()


@pytest.mark.parametrize("run_id", ["../escape", "nested/run", "", "bad value"])
def test_artifact_store_rejects_unsafe_run_ids(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(ConfigurationError):
        ArtifactStore(tmp_path, run_id)


def test_artifact_store_writes_json_yaml_and_jsonl(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, "run-001")
    store.create()

    store.write_json("manifest.json", {"run_id": "run-001"})
    store.write_yaml("config.resolved.yaml", {"seed": 42})
    store.append_jsonl("requests.jsonl", [{"sequence": 1}, {"sequence": 2}])

    assert json.loads(store.path("manifest.json").read_text()) == {"run_id": "run-001"}
    lines = store.path("requests.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in lines] == [{"sequence": 1}, {"sequence": 2}]


def test_artifact_store_rejects_nested_artifact_name(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, "run-001")

    with pytest.raises(ConfigurationError):
        store.path("logs/output.log")
