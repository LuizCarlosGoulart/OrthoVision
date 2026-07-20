# Repository layout

Per-directory responsibility, contents, dependencies, and reason to exist. The
extension axis is `manifests/` (canonical record) + `src/orthovision/labels/`
(per-source mappers): adding a dataset or specialty touches only those, not the
training loop.

| Directory | Responsibility | Expected contents | Depends on | Why it exists |
|-----------|----------------|-------------------|------------|---------------|
| `configs/` | Single source of truth for params/paths | Declarative YAMLs | consumed by train/eval | Reproducibility needs declarative, versioned config separate from code |
| `data/raw/` | Immutable copy of sources + checksum + license | DENTEX originals | — | Lets the whole pipeline be rebuilt; never edited |
| `data/interim/` | Intermediate artifacts | Normalized images, dedup output | `raw/` | Isolates expensive, re-runnable steps from raw |
| `data/processed/` | Final training-ready views | Shards + indices | `interim/`, `manifests/` | Stable boundary between data engineering and training |
| `manifests/` | Versioned JSONL canonical records + splits | `train/val/test.jsonl`, ingestion/validation reports | `split/` | Decouples logical identity from image bytes; git-versionable |
| `src/orthovision/ingest/` | Deterministic download + provenance | DENTEX/TDD connectors | — | Makes acquisition auditable and repeatable |
| `src/orthovision/validate/` | Conformance rules | Validators + reports | `ingest` | Fail early; stops garbage propagating |
| `src/orthovision/preprocess/` | Image normalization | Deterministic transforms | `validate` | Standardizes input to the backbones |
| `src/orthovision/labels/` | Taxonomy standardization | Per-source mappers | `validate` | Centralizes the label-conflict logic |
| `src/orthovision/dedup/` | pHash intra/inter | Deduplicator | `preprocess`, `labels` | Prerequisite for leakage-free split |
| `src/orthovision/split/` | Patient-aware stratified partition | Splitter + fixed seeds | `dedup` | Single point guaranteeing partition integrity |
| `src/orthovision/datamodule/` | Loading, augmentation, feature cache | Datasets/dataloaders | `manifests`, `processed` | Encapsulates frozen-vs-trainable (cache) difference |
| `src/orthovision/models/` | Backbone wrappers + adaptation modules | CLIP/BiomedCLIP, linear probe, CoOp, LoRA | — | Strategy swap without touching the training loop |
| `src/orthovision/train/` | Training/optimization loop | Trainers, schedulers | `datamodule`, `models` | Separates training from data and model |
| `src/orthovision/eval/` | Metrics + version comparison | Evaluators, calibration/OOD tests | `models`, `datamodule` | Evaluation as a first-class component |
| `src/orthovision/inference/` | Serving | Classification/retrieval API | `models` | Stable production boundary |
| `experiments/` | Per-run outputs | run_id subdirs (ckpt, logs, metrics) | `train`, `eval` | Traceability; not raw-versioned in git |
| `tests/` | Pipeline invariants | Leakage/schema tests | all of `src/` | Guarantees critical invariants mechanically |
| `scripts/` | Thin CLI entrypoints | Wrappers over `src/` | `configs/` | Keeps logic out of scripts |
| `docs/` | Architecture memory | ADD + ADRs | — | Records design and decisions |
