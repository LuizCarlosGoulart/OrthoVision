"""Configuration loading and path resolution.

All params/paths live in ``configs/*.yaml`` (single source of truth). Nothing
downstream hardcodes a path; everything resolves through here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def repo_root(start: Path | None = None) -> Path:
    """Return the repository root (nearest ancestor containing pyproject.toml)."""
    p = (start or Path(__file__)).resolve()
    for parent in (p, *p.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("repository root (pyproject.toml) not found")


def config_dir() -> Path:
    return repo_root() / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_config(name: str) -> dict[str, Any]:
    """Load a config by logical name, e.g. ``"ingest"`` or ``"data/dentex"``."""
    return load_yaml(config_dir() / f"{name}.yaml")


def paths() -> dict[str, str]:
    return load_config("paths")["paths"]


def resolve_path(rel: str | Path) -> Path:
    """Resolve a config path to absolute, relative to the repo root if not absolute."""
    p = Path(rel)
    return p if p.is_absolute() else repo_root() / p
