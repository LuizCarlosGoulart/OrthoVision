"""Assign the F2 train/val/test split over the deduplicated canonical set.

Groups records by ``patient_group_id`` (so all of a patient's images share a
fold), stratifies the groups by pathology co-occurrence, then writes one manifest
per fold with the ``split`` field filled in. Deterministic (fixed seed) => the
test fold is frozen and reproducible.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from ..config import load_config, resolve_path
from ..dedup.build import DEDUP_MANIFEST
from ..labels.schema import CanonicalRecord, read_canonical, write_canonical
from .stratify import iterative_stratify

SPLIT_HASH = "manifests/split.hash.json"


def _group_labels(records: list[CanonicalRecord]) -> tuple[dict[str, list[CanonicalRecord]], dict[str, set[str]]]:
    """Group records by patient_group_id and union their positive labels."""
    groups: dict[str, list[CanonicalRecord]] = defaultdict(list)
    labels: dict[str, set[str]] = defaultdict(set)
    for r in records:
        groups[r.patient_group_id].append(r)
        labels[r.patient_group_id] |= {k for k, on in r.labels.items() if on}
    return groups, labels


def run_split() -> tuple[dict[str, Path], dict]:
    cfg = load_config("split")["split"]
    records = read_canonical(resolve_path(DEDUP_MANIFEST))
    groups, group_labels = _group_labels(records)

    assignment = iterative_stratify(
        items=sorted(groups),
        label_sets=group_labels,
        ratios=cfg["ratios"],
        seed=cfg["seed"],
    )

    # materialize per-fold records with the split field set
    per_fold: dict[str, list[CanonicalRecord]] = {f: [] for f in cfg["ratios"]}
    for gid, recs in groups.items():
        fold = assignment[gid]
        for r in recs:
            per_fold[fold].append(dataclasses.replace(r, split=fold))

    outputs: dict[str, Path] = {}
    for fold, recs in per_fold.items():
        outputs[fold] = write_canonical(recs, resolve_path(cfg["outputs"][fold]))

    # provenance: hash of the (record_id -> fold) assignment
    record_assignment = {
        r.record_id: fold for fold, recs in per_fold.items() for r in recs
    }
    payload = json.dumps(record_assignment, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    summary = {
        "seed": cfg["seed"],
        "ratios": cfg["ratios"],
        "counts": {f: len(r) for f, r in per_fold.items()},
        "assignment_sha256": digest,
    }
    with open(resolve_path(cfg["outputs"]["assignment_hash"]), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=2)

    return outputs, summary
