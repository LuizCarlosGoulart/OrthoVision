# Architecture Design Document — OrthoVision

Fine-tuning CLIP for diagnosis on dental panoramic radiographs.

**Fixed scope:** backbones — generic CLIP (ViT-B/16) as reference, BiomedCLIP as
domain base. Datasets — DENTEX (primary) and Tufts Dental Database / TDD
(secondary). Task — multi-label over the 4 DENTEX pathologies (caries, deep
caries, periapical lesion, impacted tooth). Central hypothesis — **global**
(impacted tooth) vs **local** (caries) signal: how well a global image encoder
captures local pathology.

> **Vision-language reframing.** This is not a text-generation project. CLIP
> aligns images and text in a shared embedding space. Concepts borrowed from LLM
> fine-tuning templates are reinterpreted: "hallucination" -> spurious pathology
> attribution and incorrect text->image retrieval; "factuality" -> agreement with
> the expert label and calibration. Q&A/articles/guidelines/protocols are out of
> scope as data (see `dataset-schema.md`).

## 1. System architecture

Data flow: sources -> versioned ingestion -> validation/cleaning -> image
normalization + label standardization to the canonical schema -> deduplication ->
patient-aware split -> two training views (supervised multi-label; contrastive
image-text, later phase) -> text tokenization + feature caching -> training /
adaptation -> evaluation -> checkpoint + metrics registry -> inference.

Step dependencies are strict: Ingestion -> Validation -> (Normalization ||
Label standardization) -> Deduplication -> Split -> View export -> Tokenization/
Cache -> Training -> Evaluation -> Registry.

Critical points:
1. Split before any augmentation and before embedding cache — main leakage vector.
2. Feature cache valid only while the encoder is frozen (linear probe / CoOp);
   invalidate for LoRA / fine-tuning.
3. Pathology imbalance drives loss and metric choice.
4. Small object (caries) in a large image — the central experimental variable.
5. Single, shared, frozen test set across both views.

```
DENTEX/TDD -> [Ingest] -> RAW (immutable, checksummed)
   -> [Validate] -> [Normalize img] || [Standardize labels]
   -> [Dedup pHash intra+inter] -> [Split patient-aware, stratified, frozen test]
   -> Supervised view (image -> multi-hot 4d) + Contrastive view (image -> text)
   -> [Tokenize + feature cache if encoder frozen]
   -> [Adapt: zero-shot | linear probe | CoOp | LoRA | FT]
   -> [Eval: AUC/mAP/F1 per class, Recall@K, ECE, OOD]
   -> [Registry: ckpt + config + data hashes] -> [Inference]
```

## 2. Repository structure

See `repository-layout.md`.

## 3. Data pipeline

Where each step happens and its invariant is defined in `repository-layout.md`
and the `configs/*.yaml` files. Leakage prevention (normative):
1. Dedup before split, including inter-dataset (DENTEX<->TDD).
2. Split by patient/image identity, never by row.
3. Split precedes augmentation and embedding cache.
4. Single test set shared by both views; no test image in contrastive training.
5. DENTEX quadrant/enumeration subsets, if used, enter the same patient-grouping
   table as the split.
6. Seeds and split hashes versioned in `manifests/`.

## 4. Dataset integration

DENTEX and TDD are largely disjoint image sets, so conflict is at the label-space
/ role level, not per-image (dedup handles physical overlap).

- **DENTEX (primary):** defines the canonical label space and the classification
  test set. Reliability `strong`. Authoritative ground truth.
- **TDD (secondary, reserved for later):** free-text descriptions for the
  contrastive view + weak auxiliary label. Reliability `weak`. Never overrides
  DENTEX or the test set.

**Precedence:** for the 4-pathology diagnostic label, **DENTEX prevails** — the
label space is defined by its expert-annotated taxonomy under CC-BY; TDD's
free-text is lossy to project onto 4 classes and would inject mapping noise.

## 5. Final dataset specification

See `dataset-schema.md`.

## 6. Fine-tuning strategy

Applicable strategies: zero-shot (baseline), linear probe, CoOp, LoRA on image
encoder, partial/full fine-tune. **Recommendation: LoRA on the BiomedCLIP image
encoder + CoOp on the text side**, with linear probe and zero-shot as mandatory
baselines. Rationale: the critical variable is perception of the local signal
(caries); frozen-encoder methods (linear probe, CoOp alone) cannot improve visual
representation of that detail, so they have a low ceiling on the class that
defines the hypothesis. LoRA adapts the encoder with enough capacity for the local
detail without the overfitting/forgetting of full/partial FT at ~1k images.
Trade-off: LoRA invalidates the feature cache and adds hyperparameters (rank,
target layers).

## 7. Evaluation

- Primary metrics: per-class and macro AUC-ROC; per-class AP/mAP; macro-F1 at an
  operating threshold; Recall@K/mAP for retrieval (contrastive view, later).
- Benchmark ladder (same frozen test set): generic CLIP zero-shot -> BiomedCLIP
  zero-shot -> BiomedCLIP linear probe -> CoOp -> LoRA+CoOp. Report global-vs-local
  stratification separately (the central result).
- "Factuality" analogue: agreement with the DENTEX expert label.
- "Hallucination" analogue: per-class false positives; calibration (ECE);
  incorrect retrieval; OOD robustness.
- Minimum approval gates: (1) macro-AUC significantly above BiomedCLIP zero-shot
  and above generic zero-shot; (2) no per-class regression vs linear probe; (3)
  caries AUC strictly above linear probe; (4) ECE no worse than zero-shot.
- Version comparison: evaluate on the frozen test set, same seeds/bootstrap; a new
  version is promoted only if it passes all gates.

## 8. Experiment organization

`run_id = date + config_hash + seed`. Each run under `experiments/<run_id>/` with
`ckpt/`, `logs/`, metrics. Datasets versioned via `manifests/*.jsonl` +
checksums; models via a registry referencing `config_hash`, `data_manifest_hash`,
backbone, strategy, metrics, gates. No run is valid without config_hash +
data_manifest_hash + seed recorded.

## 9. Scalability

Extension axis: `manifests/` (canonical record) + `src/orthovision/labels/`
(per-source mappers). New dataset -> new mapper only. New specialty -> `specialty`
stops being constant, label space namespaced per specialty via `configs/`; no
directory or training-loop change. New modality -> `modality` gains a branch in
`preprocess/`. Structural limit: above tens of thousands of records, migrate JSONL
+ files to columnar/WebDataset — the only foreseen structural change.

## 10. Roadmap

| Phase | Objective | Deliverables | Depends on | Done when |
|-------|-----------|--------------|------------|-----------|
| F0 | Data foundation | ingest/validate, immutable raw with checksums | TDD access (v1 uses DENTEX only) | 100% images hashed + validation report; raw immutable |
| F1 | Canonicalization | labels/preprocess/dedup, schema v1, dedup'd manifest | F0 | Every record schema-conformant; dedup done |
| F2 | Split & views | patient-aware split, train/val/test manifests, both views | F1 | Leakage test passes; test set frozen |
| F3 | Baselines | zero-shot (generic + BiomedCLIP) + linear probe | F2 | Baselines with CI; generic-vs-BiomedCLIP table |
| F4 | Adaptation | LoRA+CoOp, versioned runs, registry | F3 | Adapted model passes all 4 gates |
| F5 | Global-vs-local analysis | stratified metrics; caries analysis | F4 | Result reported; gate 3 decided objectively |
| F6 | Inference & consolidation | inference API, ADRs, final comparison report | F4/F5 | Inference reproduces registry metrics from data+config+seed |

Critical path F0->F1->F2 is blocking. Per ADR 0001, v1 is DENTEX-only; the
contrastive view (TDD text) is a post-F4 phase, non-blocking.
