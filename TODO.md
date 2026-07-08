# CalVis — VLM Attribution TODO

## Current State (2026-03-23)
- **Working**: Word-level attention visualization using all-layers average
- **Working**: Subword→word aggregation (Inseq-style MAX across subwords)
- **Working**: Word strength color-coding in text panel
- **Working**: Attention Rollout as dropdown option (demonstrates attention sink problem)
- **Working**: BEV attribution panel (LSS + TPVFormer backends)
- **Working**: 14 GradCAM methods wired up in backend (hidden from VLM UI — not suitable for early-fusion)

---

## Attribution Methods

### Available (VLM panel)

| Method | How | Speed | Notes |
|--------|-----|-------|-------|
| All-Layers Avg | Mean attention across 30 layers × 9 heads | ~1s | **Default.** Signal reinforces, noise cancels. Per-token spatial discrimination works well. |
| Attention Rollout | Matrix-multiply attention across layers + residual | ~1s | **Demo only.** Collapses to BOS attention sink in 30-layer models. All words produce identical heatmap focused on BOS. Educational. |
| GradCAM (14 variants) | Gradient × activation on connector projection | ~2.5s | **Backend only.** Connector is a single Linear layer — no spatial richness for CAM. |

### Why Rollout Fails (Attention Sink)

Rollout multiplies attention matrices: `R = A₃₀ × A₂₉ × ... × A₁`. If any layer puts >50% attention on BOS (the attention sink), this compounds exponentially across 30 layers: 0.5³⁰ ≈ 1e-9 of original signal survives. The BOS column absorbs everything.

Averaging works because it's additive — 30 independent "votes". BOS-heavy layers get outvoted by content-attending layers.

---

## Planned: Multi-Token / Phrase Attribution

### Problem
Tokens like "traffic light" (2 tokens) or "traffic sign" (2 tokens) represent a single concept. Per-token attribution doesn't capture phrase-level grounding.

### Approaches (ranked by priority)

1. **Chefer et al. (2021) — Gradient-Weighted Relevancy Propagation**
   - Multiplies gradients × attention weights, clamps negatives, accumulates relevancy per layer
   - Token-specific, handles multi-token by propagating from each token and aggregating
   - Cost: ~2x forward pass (one forward + one backward)
   - Implementation: [hila-chefer/Transformer-MM-Explainability](https://github.com/hila-chefer/Transformer-MM-Explainability)
   - **Recommended as next step** — principled, well-cited, open-source

2. **AttnLRP (Achtibat et al., ICML 2024) — Exact LRP for Attention**
   - Mathematically exact Layer-wise Relevance Propagation rules for softmax attention
   - More faithful than Chefer's gradient approximation
   - Cost: ~1x backward pass
   - Implementation: [rachtibat/LRP-eXplains-Transformers](https://github.com/rachtibat/LRP-eXplains-Transformers)
   - License: BSD-3 personal/scientific, patented for commercial

3. **Inseq-style Span Aggregation**
   - After computing per-token attributions (any method), aggregate across phrase:
     - SubwordAggregator: merge subwords into words
     - ContiguousSpanAggregator: merge words into phrases
     - Default: **max absolute value** across span
   - Implementation: [inseq-team/inseq](https://github.com/inseq-team/inseq)

4. **FlashTrace (Feb 2026) — Span-Wise Aggregation**
   - Exploits linearity of attention to attribute entire output spans in one pass
   - 130x faster than naive per-token attribution at 5K tokens
   - No public code yet

---

## Planned: Per-Layer Visualization

- [ ] **Per-layer scrubber** — Slider (0–29) showing attention at each layer. Backend cache exists (`_per_layer_attn`). Individual layers are noisy but show interesting patterns: shallow = spatial, deep = semantic. Need global normalization across layers.

- [ ] **Gradient-weighted layer averaging** — Weight each layer by its output gradient norm for the target token. Principled layer selection without scrubbing.

---

## Planned: UI Improvements

- [ ] **Reverse direction** — Click image region → highlight which words attended to it (transpose of current word→image flow)

- [ ] **Side-by-side comparison** — Two heatmaps for "avg" vs "rollout", or two different tokens

- [ ] **Attention entropy display** — Per-word entropy bar. Low = focused, high = diffuse/uncertain.

- [ ] **Multi-word selection** — Shift+click to select phrases. Needs single-forward-pass extraction for all tokens (current `compute_word_attentions` already does this).

---

## Planned: Model Upgrades

- [ ] **Larger VLM** — Qwen2.5-VL-3B or SmolVLM2-2.2B for better visual grounding. Current 256M model hallucinates specific objects.

- [ ] **do_image_splitting=True** — 832 image tokens instead of 64. ~13× finer spatial resolution (8×8 → ~26×32) but 13× more compute.

---

## Future: Tier 2 — Full VLA Pipeline Attribution

For VLA models with flow-matching decoders:
- **Tier 1** (current): VLM reasoning attribution — which image regions influenced each generated word
- **Tier 2** (future): Perturbation-based — mask image regions, re-run full pipeline (VLM + flow matching), compare trajectory deviation. Model-agnostic, no gradients, but slow.

---

## Future: GradCAM for Other Tasks

14 pytorch-grad-cam methods wired up in `pipeline/vlm/model.py` but hidden from VLM UI. Re-expose when adding:
- CNN-based BEV model attribution (GradCAM's natural domain)
- Cross-attention VLM support (Flamingo, BLIP-2 style where GradCAM on Q-Former is meaningful)

Available: GradCAM, HiResCAM, GradCAM++, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, ScoreCAM, AblationCAM, FinerCAM, ShapleyCAM, KPCA-CAM, GradCAM-EW, RandomCAM.

---

## References

- Abnar & Zuidema (2020) — "Quantifying Attention Flow in Transformers" (Attention Rollout)
- Chefer et al. (2021) — "Generic Attention-model Explainability" (Relevancy Propagation)
- Achtibat et al. (2024) — "AttnLRP: Attention-Aware Layer-wise Relevance Propagation" (ICML 2024)
- Sarti et al. (2023) — "Inseq: An Interpretability Toolkit for Sequence Generation Models"
- jacobgil/pytorch-grad-cam — 18+ CAM methods, ViT support, HuggingFace integration
