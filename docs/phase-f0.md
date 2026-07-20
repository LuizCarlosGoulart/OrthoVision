# Phase F0 — Data foundation

**Objective:** deterministic ingestion of DENTEX into an immutable, checksummed
raw store, with a conformance validation report. DENTEX only (ADR 0001).

## State

Scaffold and implementation done.

Scaffold — structure and declarative configuration:

- Repository skeleton (`src/orthovision/*`, `configs/`, `data/`, `manifests/`,
  `docs/`, `tests/`, `scripts/`, `experiments/`).
- Declarative configs: `paths`, `data/dentex`, `data/tdd` (reserved), `ingest`,
  `validate`, `preprocess`, `dedup`, `split`, `model/{biomedclip,clip_vitb16}`,
  `experiment/base`.
- Docs: ADD (`architecture.md`), repository layout, canonical schema, ADR 0001.

Implementation — ingestion + validation:

- `orthovision.config` — YAML config loading and repo-root path resolution.
- `orthovision.hashing.sha256_file` — streaming content hash.
- `orthovision.ingest.store` — immutable raw-store ingestion; idempotent on
  unchanged sources, raises `ImmutabilityError` on changed content (unless
  `--overwrite`). DENTEX ships images inside zips, so ingestion is zip-aware
  (`ingest_archives`): it extracts only the members selected by the `archives`
  map in `configs/data/dentex.yaml`, dropping `unlabelled/` and notebook
  checkpoints, and lays files out at `<split>/<subset>/[annotations/]<basename>`.
- `orthovision.ingest.manifest` — deterministic JSONL manifest (records sorted by
  `file`) with `{file, sha256, bytes, source, license}`.
- `orthovision.ingest.dentex` — Hugging Face snapshot download (lazy import) +
  `run_ingest`. `scripts/ingest_dentex.py` is the CLI.
- `orthovision.validate.rules` / `report` — image conformance (format, readable,
  resolution) + JSON report with per-reason counts. `scripts/validate_dentex.py`
  is the CLI.
- `tests/` — 12 tests covering config load, hashing, ingest (hash coverage,
  deterministic manifest, immutability) and validation. All green
  (`python -m pytest`).

## How to run

```
pip install -e ".[dev]"
python scripts/ingest_dentex.py      # downloads DENTEX (~11.8GB), extracts zips, ingests
python scripts/validate_dentex.py    # writes manifests/validation.dentex.json
python -m pytest                     # offline; no network needed
```

## Result on real data

Ingested 2585 files (2332 images + 253 annotations). Validation: 2332 images,
2332 passed, 0 excluded. Diagnosis subset (v1 target): 705 train images + COCO
annotation, 250 test images + per-image labels, 50 val images (challenge holdout,
no labels). Auxiliary reserved subsets: quadrant (693), quadrant_enumeration
(634). `unlabelled` (1571) and notebook checkpoints dropped by policy.

## Completion criteria (from the ADD roadmap)

- 100% of DENTEX images recorded with a sha256 and a validation report.
- Raw store is immutable (re-ingest goes to a new versioned path, never overwrite).
- Ingestion is deterministic: same source revision -> identical ingestion manifest.
