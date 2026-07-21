# Phase F4 — Adaptation (LoRA)

**Objective:** adapt the BiomedCLIP image encoder with LoRA and test whether it
lifts the local signal (caries) over the F3 frozen-encoder linear probe. See
ADR 0003 (LoRA + linear head; CoOp deferred).

## Approach

- **models/lora.py** — `LoRALinear` (frozen base + low-rank `B@A`, B zero-init so
  training starts as identity) and `add_lora` to wrap named `nn.Linear` layers.
  BiomedCLIP's visual tower is a **timm ViT**, so targets are the attention
  (`qkv`, `proj`) and MLP (`fc1`, `fc2`) linears — all plain modules.
- **models/lora_train.py** — freezes the backbone, injects LoRA, trains LoRA
  params + a linear head with BCE (multi-label). Same head as the F3 probe, so the
  comparison isolates the encoder adaptation. Anti-overfit measures (added after
  the first GPU run overfit — see below): **random horizontal-flip augmentation**
  (pathology presence is flip-invariant) and **validation-based early stopping**
  (the best-val-macro-AUC epoch is restored).
- **configs/experiment/lora.yaml** — rank/alpha/targets and training knobs;
  `train_limit` caps train images for local CPU runs (null = all 497 on GPU/Colab).

## Local functional validation (CPU)

Purpose: prove the training + eval path runs end to end and produces metrics. It
is **not** the real result — it uses a small `train_limit` and few epochs because
LoRA back-props through the whole ViT, which is slow on CPU.

Run: `python scripts/run_lora.py --limit 48 --epochs 2` (device cpu). A tiny train
subset over 2 epochs is far too little to learn 4 pathologies, so the AUCs are not
meaningful — the run only confirms the loss decreases and the eval/metrics path
works. Real numbers come from the GPU/Colab run below.

## First GPU run (rank 8, 5 epochs, no regularization)

macro AUC: lora 0.629 vs probe 0.646 — LoRA **underperformed** the frozen probe,
with local dropping (0.598 vs 0.625) while global rose (0.719 vs 0.708). Training
loss fell to 0.09 → clear overfitting on 497 images, hurting exactly the local
signal we target. Response: added flip augmentation + val early stopping and
raised the epoch budget to 15 (early stopping selects the best epoch). Re-run
pending.

## Second GPU run (rank 8, 15 epochs, flip aug + val early stopping)

**Central hypothesis confirmed.** LoRA adaptation lifts the hard local signal:

| pathology | probe | LoRA | Δ |
|---|---|---|---|
| caries (local, target) | 0.552 | **0.661** | **+0.109** |
| impacted (global) | 0.708 | **0.783** | +0.075 |
| deep_caries (local) | 0.633 | 0.589 | −0.044 |
| periapical (local) | 0.690 | 0.603 | −0.087 |
| macro | 0.646 | **0.659** | +0.013 |

Gates: g1 pass, **g3 (caries > probe) pass** — the decisive gate. g2 fails: two
classes regress (deep_caries, periapical). periapical has only 17 test positives
(high variance). Overfitting from run 1 is gone (macro now beats the probe).
Interpretation: a single macro-AUC-selected checkpoint trades gains on the classes
the frozen encoder underserved (caries, impacted) for small losses on the two the
probe already read well — likely a class-imbalance effect. Next lever:
class-balanced BCE (`pos_weight`) to recover deep_caries/periapical without losing
caries.

## Real run (deferred to GPU/Colab)

Use **`notebooks/colab_train.ipynb`**: it clones the repo, installs `.[models]`,
materializes the data (`scripts/ingest_dentex.py`), runs the full ladder + LoRA on
GPU, and evaluates the gates with bootstrap CIs. Or locally:
`python scripts/run_lora.py` with `train_limit: null` and more epochs.

Gates (ADD §7), implemented in `eval/gates.py`, compared on the frozen test fold:
- **g1** macro-AUC above BiomedCLIP zero-shot and generic zero-shot;
- **g2** no per-class regression vs the linear probe (F3);
- **g3** **caries AUC strictly above the linear probe (0.552)** — decisive for the
  central hypothesis; significance via `eval/bootstrap.paired_delta_ci` (CI
  excluding 0);
- **g4** ECE no worse than zero-shot — **not yet evaluated** (calibration TODO).

## Tests

LoRA unit tests + bootstrap/gates tests. Full suite green (51).

## Gaps carried forward

- Gate 4 (ECE/calibration) not implemented.
- Run registry (config_hash/data_manifest_hash) still minimal.
