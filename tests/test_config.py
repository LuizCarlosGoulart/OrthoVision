from orthovision.config import load_config, paths, repo_root, resolve_path


def test_repo_root_has_pyproject():
    assert (repo_root() / "pyproject.toml").is_file()


def test_load_configs():
    assert load_config("ingest")["ingest"]["checksum"] == "sha256"
    assert load_config("data/dentex")["dataset"]["name"] == "dentex"
    assert "dentex_raw" in paths()


def test_resolve_relative_is_absolute():
    assert resolve_path("data/raw").is_absolute()
