# Canonical dataset schema

The canonical unit is **one record per image**, stored as a line in a JSONL
manifest under `manifests/`. Image bytes live in the data store and are referenced
by path + hash. This is the contract every downstream component reads.

Format choice: JSONL + image files (vs. relational DB or columnar/parquet). JSONL
is diffable, git-versionable at the manifest level, and streamable by the
dataloader — sufficient at this scale (~2k images). Columnar is only justified
above tens of thousands of records (see `docs/architecture.md` §9).

## Fields

| Field | Required | Description |
|-------|----------|-------------|
| `record_id` | yes | Stable record identifier |
| `image_path` | yes | Path relative to the data store |
| `image_sha256` | yes | Hash of the image bytes (dedup / traceability) |
| `patient_group_id` | yes | Grouping key for the split (all of a patient's images share it) |
| `source` | yes | `dentex` \| `tdd` |
| `modality` | yes | Fixed: `panoramic_xray` |
| `specialty` | yes | Fixed in v1: `dentistry` (field exists for future growth) |
| `language` | yes | Text language: `en` |
| `labels` | yes (supervised view) | Multi-hot over `[caries, deep_caries, periapical_lesion, impacted_tooth]` |
| `label_certainty` | yes | Per class: `certain` \| `uncertain` \| `absent` |
| `pathology_granularity` | yes | Per positive class: `global` \| `local` (supports the central hypothesis) |
| `text_description` | conditional | Clinical description; required for the contrastive view, absent => excluded from it |
| `original_taxonomy` | yes | Raw source labels (audit) |
| `reliability` | yes | `strong` (DENTEX diagnosis) \| `weak` (TDD mapping) |
| `split` | yes | `train` \| `val` \| `test` |
| `ingest_version` | yes | Pipeline version that produced the record |

Minimum metadata (record rejected at validation if missing): `source`,
`image_sha256`, `ingest_version`, `reliability`, `split`.

## Content-type representation

The project object is panoramic images, not text corpora. Content types are mapped
to valid vision-language analogues; inapplicable ones are explicitly out of scope.

- **Clinical case (image + description)** -> canonical record with
  `text_description` set (primary TDD use, later phase).
- **Structured diagnostic label** -> `labels` + `label_certainty` (primary DENTEX
  use, v1).
- **Q&A / articles / guidelines / protocols** -> **not represented as data**. They
  do not exist in the approved datasets and would turn the project into clinical
  NLP. If ever needed (e.g. a textual definition of "deep caries" for zero-shot),
  they enter as versioned **prompt templates** in `configs/`, never as image
  records.

## Example record (illustrative)

```json
{
  "record_id": "dentex_000123",
  "image_path": "data/processed/dentex/000123.png",
  "image_sha256": "…",
  "patient_group_id": "dentex_p000123",
  "source": "dentex",
  "modality": "panoramic_xray",
  "specialty": "dentistry",
  "language": "en",
  "labels": {"caries": 1, "deep_caries": 0, "periapical_lesion": 0, "impacted_tooth": 1},
  "label_certainty": {"caries": "certain", "impacted_tooth": "certain"},
  "pathology_granularity": {"caries": "local", "impacted_tooth": "global"},
  "text_description": null,
  "original_taxonomy": {"diagnosis": ["Caries", "Impacted"]},
  "reliability": "strong",
  "split": "train",
  "ingest_version": "0.0.0"
}
```
