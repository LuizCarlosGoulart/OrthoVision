# Phase F2 — Split & views

**Objective:** partition the deduplicated canonical set into leakage-free
train/val/test folds and expose the training-ready views.

## State — complete

- **split/stratify.py** — iterative multi-label stratification (Sechidis 2011),
  dependency-free. All-negative items are stratified via a virtual NONE label.
- **split/build.py** — groups records by `patient_group_id`, unions each group's
  positive labels, stratifies the groups, and writes one manifest per fold with
  the `split` field set. Writes `manifests/{train,val,test}.jsonl` and
  `manifests/split.hash.json` (seed, ratios, counts, assignment sha256).
- **scripts/split_canonical.py** — CLI + per-fold positives report.

## Result on real data

train 497 / val 105 / test 103 (~70/15/15). Rare class stays proportional across
folds — periapical_lesion 81 / 18 / 17; all 4 classes present in every fold. The
split is deterministic (fixed seed 1337): re-running yields the same assignment
sha256, so the **test fold is frozen** by construction.

## Views

The per-fold manifests are the training-ready views at record level: each row
references the raw image plus labels and split. Image normalization
(`preprocess.transform`) is applied **on the fly by the datamodule (F3)**, not
materialized to `data/processed/` in F2 — deliberately, because preprocessing
(intensity window, resize/crop) is an experiment variable (ADD §1) and baking it
into files now would freeze choices we intend to vary. Materialization can be
added later as a caching optimization without changing the manifests.

## Tests

33 total, green. New: stratification (assignment coverage, determinism, rare-label
presence, proportions) and split integrity (patient grouping, no group spans two
folds — the leakage invariant).

## Completion criteria (roadmap)

- Leakage test passes — met (no patient_group_id spans two folds).
- Test set frozen — met (deterministic assignment; stable sha256).

## Carried into F3

- Baselines consume `manifests/{train,val,test}.jsonl`.
- Datamodule applies `preprocess.transform` on the fly and tokenizes prompts.
- Feature cache is valid only for frozen-encoder strategies (zero-shot, linear
  probe, CoOp); it must be disabled for LoRA.
