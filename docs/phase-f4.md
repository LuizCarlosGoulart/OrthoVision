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
  comparison isolates the encoder adaptation.
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

## Real run (deferred to GPU/Colab)

`python scripts/run_lora.py` with `train_limit: null`, more epochs. Compared
against F3 baselines on the frozen test fold, per the ADD §7 gates:
- macro-AUC above BiomedCLIP zero-shot and generic zero-shot;
- no per-class regression vs the linear probe (F3);
- **caries AUC strictly above the linear probe (0.552)** — the decisive gate for
  the central hypothesis;
- ECE no worse than zero-shot.

## Tests

LoRA unit tests (identity init, only A/B trainable, target wrapping). Full suite
green.

## Gaps carried forward

- Bootstrap CIs for the gates (ADD §7) not yet computed.
- Run registry (config_hash/data_manifest_hash) still minimal.
