# ADR 0003 — F4 adaptation = LoRA(image encoder) + linear head; CoOp deferred

- **Status:** Accepted
- **Date:** 2026-07-21
- **Relates to:** ADD §6 (recommended LoRA + CoOp), ADR 0001

## Context

ADD §6 recommended "LoRA on the image encoder + CoOp on the text side". Two facts
change the cost/benefit for the supervised diagnosis task:

1. The central question is whether adapting the **image encoder** lifts the local
   signal (caries). The F3 baselines confirmed the gap: frozen-encoder linear
   probe reaches global 0.708 but caries only 0.552.
2. CoOp tunes the **text** side. By the ADD's own argument it cannot improve the
   encoder's visual perception of a small local lesion. Moreover BiomedCLIP's text
   tower is an HF PubMedBERT (not the standard CLIP text transformer), which makes
   a correct CoOp implementation heavy.

## Decision

F4 adapts the **image encoder with LoRA** and trains a **linear classification
head** with BCE (multi-label). CoOp is **deferred**. The supervised classifier is
the head, not text-prompt similarity, so the architecture for this task is
image-encoder + head.

## Rationale

- LoRA + head is directly comparable to the F3 linear probe (same head, frozen vs
  adapted encoder), so the experiment isolates exactly the LoRA contribution — the
  quantity the central hypothesis is about.
- Adding CoOp on top of a linear head is redundant: once the head replaces the
  text-similarity classifier, tuning text prompts no longer affects the decision.
- LoRA targets the BiomedCLIP visual tower, which is a **timm ViT**: its
  attention (`qkv`, `proj`) and MLP (`fc1`, `fc2`) are plain `nn.Linear` called as
  modules, so all are wrapped. (Note: the generic-CLIP ViT instead uses packed
  `nn.MultiheadAttention`, whose projections are accessed by `.weight` and cannot
  be wrapped the same way — but F4 adapts BiomedCLIP, not generic CLIP.)

## Trade-offs

- We drop the text-side adaptation for v1 supervised classification. This is
  acceptable because the contrastive/text path is a separate, deferred phase
  (ADR 0001) with its own dataset (TDD).
- If a later phase wants prompt-based zero-shot after adaptation, CoOp can be
  revisited on the contrastive path without changing this decision.

## Consequences

- `models/lora.py` + `models/lora_train.py` implement LoRA + head.
- Evaluation compares `biomedclip::lora` against `biomedclip::linear_probe`
  (F3) on the same frozen test fold, per the ADD §7 gates.
