# Phase F0 — Data foundation

**Objective:** deterministic ingestion of DENTEX into an immutable, checksummed
raw store, with a conformance validation report. DENTEX only (ADR 0001).

## Current state (scaffold)

Done in this step — structure and declarative configuration, no pipeline logic:

- Repository skeleton (`src/orthovision/*`, `configs/`, `data/`, `manifests/`,
  `docs/`, `tests/`, `scripts/`, `experiments/`).
- Declarative configs: `paths`, `data/dentex`, `data/tdd` (reserved), `ingest`,
  `validate`, `preprocess`, `dedup`, `split`, `model/{biomedclip,clip_vitb16}`,
  `experiment/base`.
- Docs: ADD (`architecture.md`), repository layout, canonical schema, ADR 0001.

## Remaining F0 work (implementation)

1. `ingest/` — download DENTEX from the configured source, write bytes to
   `data/raw/dentex` once, compute sha256 per file, emit
   `manifests/ingestion.dentex.jsonl` with `{file, sha256, bytes, source,
   license}`. Hard-fail on checksum mismatch on re-download.
2. `validate/` — apply `configs/validate.yaml` rules, emit
   `manifests/validation.dentex.json` with pass / excluded-by-rule counts.
3. `tests/` — assert raw immutability (no overwrite) and that every ingested file
   appears in the ingestion manifest with a hash.

## Completion criteria (from the ADD roadmap)

- 100% of DENTEX images recorded with a sha256 and a validation report.
- Raw store is immutable (re-ingest goes to a new versioned path, never overwrite).
- Ingestion is deterministic: same source revision -> identical ingestion manifest.
