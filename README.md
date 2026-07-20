# OdontoCLIP

Fine-tuning CLIP for diagnosis on dental panoramic radiographs.

Backbones: generic CLIP (ViT-B/16) as reference, BiomedCLIP as domain base.
Task (v1): multi-label classification of 4 DENTEX pathologies — caries, deep
caries, periapical lesion, impacted tooth. Central hypothesis: how well a global
image encoder captures a **local** signal (caries) vs a **global** one (impacted
tooth).

## Status

Phase **F0 — Data foundation** (scaffold + declarative configuration). No pipeline
logic implemented yet.

## Scope decisions

- v1 is **DENTEX-only**; the TDD dataset and the image-text contrastive view are
  deferred to a later phase — see [docs/adr/0001-v1-dentex-only.md](docs/adr/0001-v1-dentex-only.md).
- All documentation is in English.

## Layout

| Path | Purpose |
|------|---------|
| `configs/` | Declarative configuration (single source of truth for params/paths) |
| `data/` | Local data store (git-ignored; versioned by manifest + checksum) |
| `manifests/` | Versioned JSONL canonical records and split assignments |
| `src/clip_dental/` | Pipeline packages (ingest → … → inference) |
| `experiments/` | Per-run outputs (checkpoints, logs, metrics); git-ignored |
| `tests/` | Pipeline invariants (leakage, schema) |
| `scripts/` | Thin CLI entrypoints |
| `docs/` | Architecture Design Document and ADRs |

See [docs/architecture.md](docs/architecture.md) for the full Architecture Design
Document and [docs/repository-layout.md](docs/repository-layout.md) for the
per-directory responsibilities.
