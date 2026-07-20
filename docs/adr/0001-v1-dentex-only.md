# ADR 0001 — v1 DENTEX-only; contrastive/TDD deferred to a later phase

- **Status:** Accepted
- **Date:** 2026-07-20
- **Project context:** Fine-tuning CLIP for diagnosis on dental panoramic
  radiographs (OrthoVision). Backbones: generic CLIP (ViT-B/16) as reference and
  BiomedCLIP as domain base. Approved datasets: DENTEX (primary) and Tufts Dental
  Database — TDD (secondary).

## Decision

The **v1** of the project will use **DENTEX only**, framed as a **multi-label**
classification task over the 4 pathologies (caries, deep caries, periapical
lesion, impacted tooth).

The **TDD** dataset and the **image-text contrastive view** are deferred to a
**later phase**, already enabled by design (schema, splitter, and canonical
record), requiring no structural change.

## Alternatives considered

1. **DENTEX-only in v1** (selected).
2. **DENTEX + TDD/contrastive already in v1.**

## Rationale

1. **External dependency off the critical path.** The core object of the project
   — 4-pathology classification and the global-vs-local hypothesis — is fully
   covered by DENTEX (CC-BY, direct download). TDD depends on form approval,
   outside the team's control; binding v1 to it puts schedule risk on an
   experiment that is complementary, not foundational.

2. **Variable isolation.** Adding the contrastive view in v1 mixes two sources of
   uncertainty (visual adaptation via LoRA and image-text alignment with a weak
   label). A failure in the evaluation gates would be unattributable.
   DENTEX-only keeps a single strong source of truth and a single task.

3. **Cost of deferral ~ zero.** The canonical schema already contains
   `text_description`, `reliability=weak`, and `source`; the splitter is
   patient-aware and shared; the canonical record does not change. TDD can be
   incorporated retroactively with no rework or migration.

4. **Consistency with the precedence rule.** TDD never alters the ground truth or
   the test set (the 4-pathology diagnostic label is defined by DENTEX).
   Therefore the primary metric and approval gates are already defined by DENTEX
   alone; a DENTEX-only v1 yields exactly the success criterion of the core task.

## Accepted trade-offs

- v1 forgoes the text-to-image experiment (Recall@K) and the "pure contrastive
  CLIP" narrative.
- In exchange: faster delivery of the result that defines the project
  (global-vs-local on caries), no coupling to external approval, and early risk
  discovery (if caries AUC fails the minimum criterion, it surfaces before any
  investment in TDD).

## Reversal condition

If TDD access is already granted, the TDD download may happen in F0 (storage is
cheap), **but v1 training remains DENTEX-only** — the data merely stays ready for
the later contrastive phase.

## Consequences

- Roadmap phases F0–F5 can proceed without TDD.
- The contrastive view becomes an additional phase (post-F4), non-blocking.
- No change to the directory structure, training loop, or model registry is
  required to incorporate TDD later.
