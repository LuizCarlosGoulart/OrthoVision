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
  `--overwrite`).
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
python scripts/ingest_dentex.py      # downloads DENTEX, then ingests
python scripts/validate_dentex.py    # writes manifests/validation.dentex.json
python -m pytest                     # offline; no network needed
```

## Completion criteria (from the ADD roadmap)

- 100% of DENTEX images recorded with a sha256 and a validation report.
- Raw store is immutable (re-ingest goes to a new versioned path, never overwrite).
- Ingestion is deterministic: same source revision -> identical ingestion manifest.
