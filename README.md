---
title: BEV & VLM Attribution
emoji: 🚗
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
tags:
  - autonomous-driving
  - bev
  - attribution
  - vlm
  - interpretability
  - gradcam
  - attention-visualization
---

# BEV & VLM Attribution Debug Tool

Interactive tool for visualizing and debugging perception models used in autonomous driving.

## Three Modes

### BEV Attribution
Click any cell on the Bird's Eye View grid to answer: **"Which camera pixels drove this prediction?"**
- **LSS (Lift-Splat-Shoot)** model projects 6 surround cameras into a BEV occupancy grid
- **4 attribution methods**: GradCAM, Integrated Gradients, Attention, Occlusion
- Heatmaps overlaid on all 6 cameras showing per-pixel contribution
- GT bounding box overlay with class tooltips on hover
- Camera coverage FOV lines on BEV grid

### VLM Reasoning
Generate text descriptions of driving scenes, then click any word to answer: **"Where was the model looking when it said this?"**
- **SmolVLM-256M** (HuggingFace, Apache 2.0) generates scene descriptions from camera images
- **Word-level attention**: subword tokens grouped into words, color-coded by attention strength
- Click a word → attention heatmap shows which image patches the LLM attended to
- All-layers average extracts attention from 30 LLM layers in a single forward pass

### VLM → BEV Projection
Project VLM attention from camera space into Bird's Eye View to see **where in the physical world** the model is attending.
- Front camera attention projected onto BEV grid using camera calibration
- LSS vehicle detections overlaid as cyan contours for comparison
- Preset prompts for analyzing closest/farthest vehicles, traffic conditions
- Click words to see per-word BEV attention projection
- Click BEV cell to see corresponding camera location highlighted

## Architecture

```
BEV Mode:
  6 Cameras → EfficientNet → Lift-Splat → BEV Grid → Click cell → Attribution → Camera heatmaps

VLM Mode:
  Camera Image → SigLIP Vision Encoder → Linear Connector → LLM (30 layers) → Text
                                                              ↑
                                          Click word → Extract self-attention → Image heatmap

VLM→BEV Mode:
  Camera heatmap → Inverse projection (K, E matrices) → BEV grid overlay + LSS comparison
```

## Data
Uses nuScenes mini dataset (10 scenes, 6 surround cameras per frame).

## Run Locally
```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:7860
```

## Built With
- FastAPI + embedded HTML5/Canvas frontend
- PyTorch, HuggingFace Transformers, Captum, pytorch-grad-cam
- SmolVLM-256M-Instruct (Apache 2.0)
- LSS (Lift-Splat-Shoot) for BEV perception
