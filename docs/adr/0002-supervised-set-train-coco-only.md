# ADR 0002 — v1 supervised set = 705 train-COCO images only

- **Status:** Accepted
- **Date:** 2026-07-20
- **Relates to:** ADR 0001 (DENTEX-only v1)

## Context

DENTEX ships two incompatible diagnosis annotation schemas:

- **Train** (`training_data/quadrant-enumeration-disease/`): a clean COCO
  annotation with per-tooth `category_id_3` in exactly the 4 canonical classes
  (Impacted, Caries, Periapical Lesion, Deep Caries). 705 images. Expert-labeled.
- **Test** (`test_data/disease/label/`): per-image LabelMe polygons whose `label`
  strings encode disease in a divergent Turkish clinical vocabulary. Observed
  tokens: çürük (caries, 747), küretaj (curettage, 265), gömülü (impacted, 221),
  kanal (root-canal, 161), saglam (healthy, 91), lezyon (lesion, 75), çekim
  (extraction, 29), kırık (fractured, 11). Roughly half of these (küretaj, kanal,
  saglam, çekim, kırık) do not correspond to any of the 4 classes, and there is
  **no token for Deep Caries**.
- **Validation** (`validation_data/`): 50 images, no labels (challenge holdout).

## Decision

The v1 canonical **supervised** set is the **705 train-COCO images only**. The F2
patient-aware split partitions these 705 into train/val/test. The packaged DENTEX
test (250) and validation (50) are **not** used as supervised ground truth.

## Alternatives considered

1. **705 train-COCO only** (selected).
2. Also map the 250 test images via a best-effort Turkish→class mapping
   (reliability=weak).
3. Split the 705, and use the 250 test images with weak labels as a secondary,
   exploratory-only evaluation set.

## Rationale

Mapping the test vocabulary onto the 4 classes is lossy and partly undefined:
~50% of tokens have no corresponding class, and Deep Caries has no token at all,
guaranteeing systematic false negatives. Injecting that as labels would violate
the precedence principle of ADR 0001 (the 4-class ground truth must stay clean and
authoritative) and would contaminate metrics with mapping noise indistinguishable
from model error. The train COCO is the only unambiguous, expert-annotated 4-class
source.

## Accepted trade-offs

- The split operates on a small set (705 images), so val/test folds are modest.
  Mitigated by the few-shot framing already in scope and by stratifying on
  pathology co-occurrence (F2).
- The 250 real DENTEX test images are not used in v1. They remain available for a
  future weak-labeled, exploratory-only evaluation (alternative 3), which a later
  ADR may adopt without changing the canonical pipeline.

## Consequences

- F1 canonicalization maps only the train COCO (`labels/dentex_coco.py`).
- F2 split input = `manifests/canonical.dentex.jsonl` (705 records).
- The reserved auxiliary subsets (quadrant, quadrant_enumeration) and the packaged
  test/val images stay in the raw store but outside the v1 supervised manifest.
