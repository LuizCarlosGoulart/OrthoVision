# Phase F1 — Canonicalization

**Objective:** turn the immutable raw store (F0) into a schema-conformant,
deduplicated canonical manifest, and provide the deterministic image
normalization used by later view export.

## State — complete

- **labels/** — `dentex_coco.py` maps the train COCO (per-tooth `category_id_3`)
  to image-level multi-hot over the 4 canonical pathologies; `schema.py` defines
  `CanonicalRecord` (pre-split; `split` assigned in F2); `build.py` +
  `scripts/build_canonical.py` write `manifests/canonical.dentex.jsonl`.
  Result: **705 canonical records**. Imbalance: caries 623, deep_caries 321,
  impacted 254, periapical 116; 27 all-negative.
  Only the train COCO is mapped — see ADR 0002 (test/val packaged labels use a
  divergent Turkish vocabulary and are excluded).
- **dedup/** — `phash.py` (256-bit dHash, PIL-only) + `dedup.py` (single-linkage
  grouping, keep strongest reliability then lowest id) + `build.py` +
  `scripts/dedup_canonical.py` write `manifests/canonical.dentex.dedup.jsonl` and
  `manifests/dedup.report.json`. Result: **705 kept, 0 removed**.
  Hash resolution was decisive: a 64-bit dHash at threshold 5 wrongly flagged 54
  distinct patients (all panoramics share global anatomy); at 256-bit the nearest
  pair is 32/256 bits apart — DENTEX train has no true (near-)duplicates.
- **preprocess/** — `transform.py`: deterministic intensity windowing
  (percentile), grayscale→RGB, aspect-preserving resize+pad to 224. Used at view
  export (F2); not materialized in F1.

## Tests

27 total, green: config, hashing, ingest (loose + archives), label mapper, dedup,
preprocess.

## How to run

```
python scripts/build_canonical.py    # -> manifests/canonical.dentex.jsonl (705)
python scripts/dedup_canonical.py     # -> canonical.dentex.dedup.jsonl (705) + report
```

## Completion criteria (roadmap)

- Every record schema-conformant — met (705 records via `CanonicalRecord`).
- Dedup done — met (256-bit dHash, 0 removed, report written).

## Carried into F2

- Split input: `manifests/canonical.dentex.dedup.jsonl` (705).
- `patient_group_id` currently equals `record_id` (1 panoramic = 1 patient
  assumption; DENTEX exposes no patient id). Revisit if patient metadata appears.
- View export applies `preprocess.transform` to materialize processed images.
