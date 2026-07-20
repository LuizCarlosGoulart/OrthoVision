"""Near-duplicate detection over canonical records (runs before the F2 split).

Groups records whose image dHashes are within ``hamming_threshold`` and keeps one
representative per group (strongest reliability, then lowest record_id). Removed
records are reported with the record they duplicate, so the split never places
equivalent images in different partitions.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..labels.schema import CanonicalRecord
from .phash import hamming

_RELIABILITY_RANK = {"strong": 0, "weak": 1}


@dataclass(frozen=True)
class DedupResult:
    kept: list[CanonicalRecord]
    removed: dict[str, str]  # removed record_id -> duplicate_of record_id
    groups: list[list[str]]  # record_ids in each duplicate group (size >= 2)


def _preferred(a: CanonicalRecord, b: CanonicalRecord) -> CanonicalRecord:
    """Return the record to keep: strongest reliability, then lowest record_id."""
    ra = _RELIABILITY_RANK.get(a.reliability, 99)
    rb = _RELIABILITY_RANK.get(b.reliability, 99)
    if ra != rb:
        return a if ra < rb else b
    return a if a.record_id <= b.record_id else b


def deduplicate(
    records: list[CanonicalRecord],
    hashes: dict[str, int],
    threshold: int,
) -> DedupResult:
    """Deduplicate ``records`` using precomputed ``{record_id: dhash}``."""
    ordered = sorted(records, key=lambda r: r.record_id)
    kept: list[CanonicalRecord] = []
    removed: dict[str, str] = {}
    group_of: dict[str, list[str]] = {}

    for rec in ordered:
        h = hashes[rec.record_id]
        match: CanonicalRecord | None = None
        for keeper in kept:
            if hamming(hashes[keeper.record_id], h) <= threshold:
                match = keeper
                break

        if match is None:
            kept.append(rec)
            continue

        # rec duplicates an already-kept record; decide which survives
        winner = _preferred(match, rec)
        loser = rec if winner.record_id == match.record_id else match
        if loser.record_id == match.record_id:
            kept.remove(match)
            kept.append(rec)
        removed[loser.record_id] = winner.record_id
        group_of.setdefault(winner.record_id, [winner.record_id]).append(loser.record_id)

    groups = [sorted(set(g)) for g in group_of.values()]
    return DedupResult(kept=sorted(kept, key=lambda r: r.record_id), removed=removed, groups=groups)
