# Phase F3 — Baselines

**Objective:** establish the baseline ladder on the frozen test fold and quantify
the domain gap and the global-vs-local signal, before any encoder adaptation (F4).

## Approach

- **backbone.py** — open_clip wrapper. Critical domain choice: the default CLIP
  preprocess does Resize+CenterCrop, which crops the sides of a wide panoramic
  (where molars/impactions are). We instead apply intensity windowing +
  grayscale→RGB + aspect-preserving pad to the model input, then normalize with
  the model's own mean/std. Encoding streams in batches, returns L2-normalized
  embeddings.
- **zero_shot.py** — per-class score = cosine(image, class-prompt embedding).
  Prompts in `configs/experiment/base.yaml:class_prompts`. AUC is rank-based, so
  no cross-class calibration is needed.
- **linear_probe.py** — one `Linear(D, 4)` with BCE over frozen features; backbone
  frozen so features are computed once per fold.
- **eval/metrics.py** — dependency-free ROC-AUC (rank-sum, tie-aware) and AP.

## Baseline ladder (macro AUC on the frozen test fold, n=103)

| model :: strategy | macro AUC | local | global |
|---|---|---|---|
| clip_vitb16 :: zero_shot | 0.551 | 0.564 | 0.513 |
| biomedclip :: zero_shot | 0.514 | 0.533 | 0.456 |
| biomedclip :: linear_probe | 0.646 | 0.625 | **0.708** |

Per-class AUC (linear probe): caries 0.552, deep_caries 0.633,
periapical_lesion 0.690, impacted_tooth 0.708.
(Linear probe trained to convergence: 1000 epochs, lr 1e-2, full-batch.)

## Findings

1. **Large domain gap.** Both backbones are near chance in zero-shot — CLIP has
   essentially no off-the-shelf ability on panoramic dental pathology.
2. **Generic CLIP ≥ BiomedCLIP in zero-shot** (0.551 vs 0.514). BiomedCLIP's
   PubMed pretraining is radiology/pathology-heavy with little panoramic dental,
   so its domain prior does not transfer here without adaptation.
3. **Global ≥ local, and caries is the weak point (central hypothesis, first
   evidence).** The linear probe reads the global signal (impacted_tooth, 0.708)
   best; among local signals caries — the smallest, subtlest lesion — is the
   hardest (0.552), below deep_caries (0.633) and periapical (0.690). A frozen
   global encoder underserves the fine local signal. This is the motivation for
   F4: adapt the image encoder (LoRA) to lift caries and the local classes.

## Reproduce

```
pip install -e ".[models]"          # torch + open_clip + transformers
python scripts/run_baselines.py     # CPU ok (~9 min); writes experiments/<run>/baselines.json
```

## Gaps carried into F4

- Provenance: baselines record seed but not yet config_hash/data_manifest_hash
  (proper run registry lands with F4 adaptation).
- Bootstrap confidence intervals for the gates (ADD §7) are not yet computed.
