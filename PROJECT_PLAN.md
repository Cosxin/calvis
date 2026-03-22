# BEV Attribution Debug Tool — Project Plan

## Overview

An interactive tool that lets you click a BEV (Bird's Eye View) grid cell and see **which image pixels contributed to that cell's prediction**, using gradient-based attribution (Captum) on open-source BEV models trained on nuScenes.

**Stack**: Python, PyTorch, Captum, Gradio, MMDet3D, nuScenes
**Data**: nuScenes mini split (free, ~4GB)
**Model**: BEVFormer-tiny or BEVDet-tiny (pretrained checkpoint)
**Timeline**: MVP in 1 weekend, polished demo in 1 week

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   Gradio UI                      │
│  ┌──────────┐  ┌─────────────────────────────┐  │
│  │ BEV Grid  │  │  6x Camera Panels           │  │
│  │ (click)   │  │  (attribution heatmaps)     │  │
│  │           │  │                             │  │
│  │  [i,j] ──────► per-camera saliency maps   │  │
│  └──────────┘  └─────────────────────────────┘  │
│  ┌──────────────────────────────────────────┐   │
│  │ Controls: method, layer, class, baseline  │   │
│  └──────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────┘
                     │
        ┌────────────▼────────────────┐
        │      Attribution Engine      │
        │  - IntegratedGradients       │
        │  - GradCAM                   │
        │  - Occlusion                 │
        │  - Attention extraction      │
        │  (Captum + custom hooks)     │
        └────────────┬────────────────┘
                     │
        ┌────────────▼────────────────┐
        │      Model Wrapper           │
        │  - Loads BEVFormer/BEVDet    │
        │  - forward_fn(images) →      │
        │    scalar at [i,j,class]     │
        │  - Hook registry for         │
        │    intermediate features     │
        └────────────┬────────────────┘
                     │
        ┌────────────▼────────────────┐
        │      Data Loader             │
        │  - nuScenes mini sample      │
        │  - Returns 6 camera images   │
        │    + calibration + GT boxes  │
        └─────────────────────────────┘
```

---

## Task Breakdown

### Agent 1 — Data & Model Pipeline

**Scope**: Everything below the UI. Load data, load model, run inference, produce a BEV grid output.

| # | Task | Output | Est |
|---|------|--------|-----|
| 1.1 | Set up environment: `mmdet3d`, `nuscenes-devkit`, `captum`, `gradio` | `requirements.txt`, install script | 1h |
| 1.2 | Write nuScenes data loader | Function: `load_sample(scene_idx) → dict(images=[6,3,H,W], intrinsics, extrinsics, gt_boxes)` | 2h |
| 1.3 | Load BEVFormer-tiny pretrained checkpoint | Function: `load_model() → model` that runs on CPU/single-GPU | 2h |
| 1.4 | Write inference wrapper | Function: `infer(model, sample) → bev_grid [C, H_bev, W_bev]` | 2h |
| 1.5 | Write model wrapper for Captum | Function: `forward_fn(images, cell_i, cell_j, class_idx) → scalar` | 2h |
| 1.6 | Verify end-to-end: load sample → infer → get BEV output → matches expected shape | Test script | 1h |

**Deliverable**: `pipeline/` module where you can do:
```python
sample = load_sample(0)
model = load_model()
bev = infer(model, sample)       # [C, 200, 200]
val = forward_fn(model, sample, i=100, j=100, cls=0)  # scalar
```

---

### Agent 2 — Attribution Engine

**Scope**: Given the model wrapper from Agent 1, compute per-pixel attributions for each camera.

| # | Task | Output | Est |
|---|------|--------|-----|
| 2.1 | Implement Integrated Gradients attribution | Function: `attr_ig(model, sample, i, j, cls) → [6, H, W]` heatmaps | 2h |
| 2.2 | Implement GradCAM at configurable layer | Function: `attr_gradcam(model, sample, i, j, cls, layer) → [6, H, W]` | 2h |
| 2.3 | Implement attention extraction (no Captum, direct hook) | Function: `attr_attention(model, sample, i, j) → [6, H, W]` | 3h |
| 2.4 | Implement occlusion-based attribution | Function: `attr_occlusion(model, sample, i, j, cls, patch_size) → [6, H, W]` | 1h |
| 2.5 | Normalize & postprocess heatmaps (abs, blur, percentile clip) | Utility functions | 1h |
| 2.6 | Benchmark: time each method, document tradeoffs | `BENCHMARKS.md` | 1h |

**Deliverable**: `attribution/` module with a unified interface:
```python
heatmaps = attribute(model, sample, i=100, j=100, cls=0, method="ig")
# heatmaps: [6, H, W] numpy arrays, normalized 0-1
```

**Key decision**: Integrated Gradients needs a baseline. Default to black images. Expose as parameter.

---

### Agent 3 — UI & Visualization

**Scope**: Gradio app. Render BEV grid, camera panels, wire clicks to attribution, display results.

| # | Task | Output | Est |
|---|------|--------|-----|
| 3.1 | BEV grid rendering: top-down grid with ego vehicle, class-colored cells from model output | Function: `render_bev(bev_grid, gt_boxes) → PIL.Image` | 2h |
| 3.2 | Camera panel rendering: overlay heatmap on camera image | Function: `render_camera(image, heatmap, name) → PIL.Image` | 1h |
| 3.3 | Click-to-cell mapping: Gradio image click → BEV cell index | Callback wiring | 2h |
| 3.4 | Main Gradio app layout: BEV on left, 6 cameras in 2x3 grid on right | `app.py` | 2h |
| 3.5 | Controls: method dropdown, class selector, layer selector, baseline toggle | Gradio components | 1h |
| 3.6 | Geometric projection overlay: also draw the classic pinhole projection marker (no attribution, just geometry) as reference | Overlay on camera panels | 1h |
| 3.7 | Precompute/cache: cache attributions for previously clicked cells | LRU cache wrapper | 1h |

**Deliverable**: `python app.py` → opens browser → click a cell → see heatmaps.

---

## Agent Dependency Graph

```
Agent 1 (Data & Model)
    │
    ├──► Agent 2 (Attribution)  ← needs forward_fn from Agent 1
    │         │
    │         ▼
    └──► Agent 3 (UI)           ← needs both Agent 1 outputs + Agent 2 heatmaps
```

- Agent 1 starts first, delivers the model wrapper interface (can stub with random tensors initially)
- Agent 2 and Agent 3 can start in parallel once Agent 1 defines the interface
- Agent 3 can develop with mock heatmaps (random noise) while Agent 2 implements real attribution
- Final integration: wire real attribution into UI callbacks

---

## File Structure

```
calvis/
├── PROJECT_PLAN.md          # this file
├── requirements.txt
├── app.py                   # Gradio entry point
├── pipeline/
│   ├── __init__.py
│   ├── data.py              # nuScenes loader
│   ├── model.py             # BEVFormer loading & inference
│   └── wrapper.py           # Captum-compatible forward_fn
├── attribution/
│   ├── __init__.py
│   ├── integrated_gradients.py
│   ├── gradcam.py
│   ├── attention.py
│   ├── occlusion.py
│   └── utils.py             # normalization, blurring, caching
├── viz/
│   ├── __init__.py
│   ├── bev.py               # BEV grid rendering
│   └── camera.py            # camera panel rendering + heatmap overlay
└── scripts/
    ├── download_nuscenes.sh  # download mini split
    ├── download_model.sh     # download pretrained checkpoint
    └── test_pipeline.py      # end-to-end sanity check
```

---

## MVP Milestones

| Milestone | Definition of Done | Target |
|-----------|-------------------|--------|
| **M0: Env** | All deps installed, nuScenes mini downloaded, model checkpoint downloaded | Day 1 morning |
| **M1: Inference** | `python scripts/test_pipeline.py` → prints BEV grid shape, no errors | Day 1 afternoon |
| **M2: Attribution** | `attribute(model, sample, 100, 100, 0, "gradcam")` → returns 6 heatmaps | Day 1 evening |
| **M3: UI** | `python app.py` → click BEV cell → see heatmaps on cameras (GradCAM only) | Day 2 morning |
| **M4: Polish** | All 4 methods working, caching, geometric overlay, class selector | Day 2 afternoon |

---

## Key Technical Risks

| Risk | Mitigation |
|------|-----------|
| BEVFormer is complex to load outside MMDet3D's runner | Use MMDet3D's `init_model()` API directly, don't try to extract the model standalone |
| Captum doesn't play well with MMDet3D's data format | Write a thin wrapper that accepts raw tensors, bypasses MMDet3D's data pipeline for the backward pass |
| Integrated Gradients is too slow for interactive use | Default to GradCAM (one backward pass). Offer IG as "detailed analysis" with a progress bar |
| Gradio image click coordinates don't map cleanly to BEV cells | Render BEV at a fixed resolution (e.g., 800x800), compute cell index from pixel coords arithmetically |
| GPU memory for backward pass through full model | Use BEVFormer-tiny (smallest variant), or run attribution on CPU with `float32` if needed |

---

## Phase 2 (Not Now)

- [ ] Temporal attribution: multi-frame models, show which past frame matters
- [ ] Load real camera images as backgrounds in camera panels
- [ ] Compare two checkpoints side-by-side
- [ ] Export attribution report (HTML or PDF)
- [ ] Deploy to HuggingFace Spaces
- [ ] LSS depth-bin decomposition view
- [ ] Precompute all-cell attributions → animate as video

---

## Quick Start (After Setup)

```bash
# 1. Environment
pip install -r requirements.txt

# 2. Data & model
bash scripts/download_nuscenes.sh   # ~4GB, mini split
bash scripts/download_model.sh      # ~200MB, BEVFormer-tiny

# 3. Verify
python scripts/test_pipeline.py

# 4. Run
python app.py
# → opens http://localhost:7860
# → click any BEV cell
# → see which image pixels the model used
```
