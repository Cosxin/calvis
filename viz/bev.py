"""BEV grid rendering for the attribution debug tool — three visualisation modes."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# nuScenes class names and colours (10 classes)
CLASS_NAMES = [
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
    'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier',
]

CLASS_COLORS = {
    'car':                  (0, 204, 255),
    'truck':                (255, 102, 0),
    'bus':                  (255, 204, 0),
    'trailer':              (204, 102, 0),
    'construction_vehicle': (255, 153, 0),
    'pedestrian':           (255, 51, 102),
    'motorcycle':           (153, 102, 255),
    'bicycle':              (102, 255, 51),
    'traffic_cone':         (255, 102, 102),
    'barrier':              (136, 136, 136),
}

CLASS_COLORS_BY_IDX = [CLASS_COLORS[name] for name in CLASS_NAMES]

BEV_IMAGE_SIZE = 800


def _auto_color(idx):
    """Generate a distinct color for class indices without a predefined color."""
    import colorsys
    hue = (idx * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.9)
    return (int(r * 255), int(g * 255), int(b * 255))


def _get_font(size=11):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/SFMono-Regular.otf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _turbo_colormap(val):
    """Attempt a crude turbo LUT: 0→blue, 0.25→cyan, 0.5→green, 0.75→yellow, 1→red."""
    v = float(np.clip(val, 0.0, 1.0))
    if v < 0.25:
        t = v / 0.25
        return (0, int(t * 255), 255)
    elif v < 0.5:
        t = (v - 0.25) / 0.25
        return (0, 255, int((1 - t) * 255))
    elif v < 0.75:
        t = (v - 0.5) / 0.25
        return (int(t * 255), 255, 0)
    else:
        t = (v - 0.75) / 0.25
        return (255, int((1 - t) * 255), 0)


def _turbo_colormap_array(arr):
    """Vectorised turbo colormap for a [H,W] float32 array in [0,1]."""
    h, w = arr.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    a = np.clip(arr, 0.0, 1.0)

    m1 = a < 0.25
    t = a[m1] / 0.25
    rgb[m1, 0] = 0;  rgb[m1, 1] = (t * 255).astype(np.uint8);  rgb[m1, 2] = 255

    m2 = (a >= 0.25) & (a < 0.5)
    t = (a[m2] - 0.25) / 0.25
    rgb[m2, 0] = 0;  rgb[m2, 1] = 255;  rgb[m2, 2] = ((1 - t) * 255).astype(np.uint8)

    m3 = (a >= 0.5) & (a < 0.75)
    t = (a[m3] - 0.5) / 0.25
    rgb[m3, 0] = (t * 255).astype(np.uint8);  rgb[m3, 1] = 255;  rgb[m3, 2] = 0

    m4 = a >= 0.75
    t = (a[m4] - 0.75) / 0.25
    rgb[m4, 0] = 255;  rgb[m4, 1] = ((1 - t) * 255).astype(np.uint8);  rgb[m4, 2] = 0

    return rgb


# ── Drawing helpers ──────────────────────────────────────────────────────────

def _draw_grid_and_axes(draw, H, W, cell_px, grid_range):
    """Grid lines, axis lines, range labels, ego icon."""
    S = BEV_IMAGE_SIZE
    center = S // 2

    # Subtle grid lines
    step = max(1, int(H / 20))
    for idx in range(0, max(H, W) + 1, step):
        p = int(idx * cell_px)
        if p < S:
            draw.line([(p, 0), (p, S)], fill=(25, 50, 55), width=1)
            draw.line([(0, p), (S, p)], fill=(25, 50, 55), width=1)

    # Axis cross
    draw.line([(center, 0), (center, S)], fill=(45, 115, 125), width=1)
    draw.line([(0, center), (S, center)], fill=(45, 115, 125), width=1)

    # Ego triangle
    cx, cy = center, center
    sz = 10
    draw.polygon([(cx, cy - sz), (cx - sz * 0.6, cy + sz * 0.6),
                  (cx + sz * 0.6, cy + sz * 0.6)], fill=(85, 204, 255))

    # Range labels
    font = _get_font(10)
    lbl = (90, 90, 90)
    for frac in [0.25, 0.5, 0.75]:
        px = int(frac * S)
        val = -grid_range + frac * 2 * grid_range
        draw.text((px + 2, center + 3), f"{val:.0f}m", fill=lbl, font=font)
        draw.text((center + 3, px + 2), f"{grid_range - frac * 2 * grid_range:.0f}m",
                  fill=lbl, font=font)
    draw.text((S - 45, center - 13), "Right →", fill=(85, 204, 255), font=font)
    draw.text((center + 4, 3), "↑ Fwd", fill=(85, 204, 255), font=font)
    draw.text((center + 4, S - 14), "↓ Back", fill=(85, 140, 180), font=font)
    draw.text((2, center - 13), "← Left", fill=(85, 140, 180), font=font)


def _draw_gt_boxes(draw, gt_boxes, grid_range):
    S = BEV_IMAGE_SIZE
    if gt_boxes is None:
        return
    for box in gt_boxes:
        cls_idx = box.get('class_idx', 0)
        color = CLASS_COLORS_BY_IDX[cls_idx] if cls_idx < len(CLASS_COLORS_BY_IDX) else (200, 200, 200)
        corners = box.get('corners')
        if corners is not None and len(corners) >= 4:
            pts = []
            for cx, cy in corners:
                px = (cx + grid_range) / (2 * grid_range) * S
                py = (grid_range - cy) / (2 * grid_range) * S
                pts.append((px, py))
            pts.append(pts[0])
            draw.line(pts, fill=color, width=2)


# ── Public API ───────────────────────────────────────────────────────────────

def render_bev(bev_grid, mode='argmax', target_class=0,
               gt_boxes=None, selected_cell=None,
               grid_range=51.2, resolution=0.512,
               class_names=None, class_colors=None):
    """Render BEV grid as a PIL image.

    Args:
        bev_grid:  [C, H, W] numpy array of class logits, or None.
        mode:      'argmax' | 'class_heatmap' | 'composite'
        target_class: int — class channel for 'class_heatmap' mode.
        gt_boxes:  list of box dicts, or None.
        selected_cell: (i, j) or None — drawn if provided (for server-rendered mode).
        grid_range: half-range of BEV in metres.
        resolution: cell size in metres.

    Returns:
        PIL.Image (RGB) 800×800.
    """
    S = BEV_IMAGE_SIZE
    img = Image.new('RGB', (S, S), color=(10, 10, 10))

    # Use provided class info or fall back to module-level defaults
    _cnames = class_names or CLASS_NAMES
    _ccolors_by_idx = (
        [class_colors[n] for n in _cnames]
        if class_colors else
        [CLASS_COLORS.get(n, _auto_color(i)) for i, n in enumerate(_cnames)]
    )

    if bev_grid is not None:
        C, H, W = bev_grid.shape
    else:
        H = W = int(2 * grid_range / resolution)
        C = 0

    cell_px = S / max(H, W)

    # ── Render prediction pixels ─────────────────────────────────────────
    if bev_grid is not None and C > 0:
        pixels = np.full((S, S, 3), 10, dtype=np.uint8)

        if mode == 'class_heatmap':
            pixels = _render_class_heatmap(bev_grid, target_class, H, W, cell_px, S)
        elif mode == 'composite':
            pixels = _render_composite(bev_grid, H, W, cell_px, S, _ccolors_by_idx)
        else:  # argmax
            pixels = _render_argmax(bev_grid, H, W, cell_px, S, _ccolors_by_idx)

        img = Image.fromarray(pixels)

    draw = ImageDraw.Draw(img)

    # Overlays
    _draw_grid_and_axes(draw, H, W, cell_px, grid_range)
    _draw_gt_boxes(draw, gt_boxes, grid_range)

    # Optional selected cell
    if selected_cell is not None:
        si, sj = selected_cell
        if 0 <= si < H and 0 <= sj < W:
            x0, y0 = int(sj * cell_px), int(si * cell_px)
            x1, y1 = int((sj + 1) * cell_px), int((si + 1) * cell_px)
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            od.rectangle([x0, y0, x1, y1], fill=(80, 220, 255, 70))
            od.rectangle([x0, y0, x1, y1], outline=(85, 204, 255), width=3)
            img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')

    return img


# ── Rendering modes ──────────────────────────────────────────────────────────

def _render_argmax(bev_grid, H, W, cell_px, S, colors_by_idx=None):
    """Each cell coloured by the top-scoring class, alpha ∝ confidence."""
    colors = colors_by_idx or CLASS_COLORS_BY_IDX
    pixels = np.full((S, S, 3), 10, dtype=np.uint8)
    class_map = np.argmax(bev_grid, axis=0)  # [H, W]
    conf_map = np.max(bev_grid, axis=0)

    C = bev_grid.shape[0]
    threshold = max(0.0, 1.0 / C) if np.all(bev_grid >= 0) else 0.0

    for i in range(H):
        for j in range(W):
            if conf_map[i, j] <= threshold:
                continue
            cls = int(class_map[i, j])
            if cls < 0 or cls >= len(colors):
                continue
            color = np.array(colors[cls], dtype=np.float32)
            alpha = float(np.clip(conf_map[i, j], 0.0, 1.0))
            alpha = max(alpha, 0.3) * 0.7

            y0, y1 = int(i * cell_px), int((i + 1) * cell_px)
            x0, x1 = int(j * cell_px), int((j + 1) * cell_px)
            y0, y1 = max(0, y0), min(S, y1)
            x0, x1 = max(0, x0), min(S, x1)

            bg = pixels[y0:y1, x0:x1].astype(np.float32)
            pixels[y0:y1, x0:x1] = (bg * (1 - alpha) + color * alpha).astype(np.uint8)

    return pixels


def _render_class_heatmap(bev_grid, class_idx, H, W, cell_px, S):
    """Show a single class channel as a continuous turbo heatmap."""
    pixels = np.full((S, S, 3), 10, dtype=np.uint8)
    C = bev_grid.shape[0]
    class_idx = max(0, min(C - 1, class_idx))

    channel = bev_grid[class_idx]  # [H, W]

    # Normalise to [0, 1] using robust percentile scaling
    lo = float(np.percentile(channel, 2))
    hi = float(np.percentile(channel, 98))
    if hi - lo < 1e-6:
        hi = lo + 1.0
    normed = (channel - lo) / (hi - lo)
    normed = np.clip(normed, 0.0, 1.0)

    # Build full-res heatmap by painting each cell
    for i in range(H):
        for j in range(W):
            val = normed[i, j]
            if val < 0.05:
                continue  # skip near-zero for clean look

            y0, y1 = int(i * cell_px), int((i + 1) * cell_px)
            x0, x1 = int(j * cell_px), int((j + 1) * cell_px)
            y0, y1 = max(0, y0), min(S, y1)
            x0, x1 = max(0, x0), min(S, x1)

            r, g, b = _turbo_colormap(val)
            alpha = max(val, 0.2) * 0.8

            bg = pixels[y0:y1, x0:x1].astype(np.float32)
            fg = np.array([r, g, b], dtype=np.float32)
            pixels[y0:y1, x0:x1] = (bg * (1 - alpha) + fg * alpha).astype(np.uint8)

    return pixels


def _render_composite(bev_grid, H, W, cell_px, S, colors_by_idx=None):
    """Blend top-3 classes at each cell, weighted by logit magnitude."""
    colors = colors_by_idx or CLASS_COLORS_BY_IDX
    pixels = np.full((S, S, 3), 10, dtype=np.uint8)
    C = bev_grid.shape[0]

    # Get top-3 classes per cell
    if C >= 3:
        top3_idx = np.argpartition(-bev_grid, 3, axis=0)[:3]  # [3, H, W]
    else:
        top3_idx = np.argsort(-bev_grid, axis=0)[:min(3, C)]

    for i in range(H):
        for j in range(W):
            vals = []
            for k in range(top3_idx.shape[0]):
                ci = int(top3_idx[k, i, j])
                logit = float(bev_grid[ci, i, j])
                if ci < len(colors):
                    vals.append((logit, colors[ci]))

            if not vals:
                continue

            # Softmax-like weighting
            logits = np.array([v[0] for v in vals], dtype=np.float64)
            logits = logits - logits.max()
            exp_l = np.exp(logits)
            weights = exp_l / (exp_l.sum() + 1e-8)

            mixed = np.zeros(3, dtype=np.float64)
            for w, (_, col) in zip(weights, vals):
                mixed += w * np.array(col, dtype=np.float64)

            total_conf = float(np.max([v[0] for v in vals]))
            alpha = float(np.clip(total_conf, 0.0, 1.0))
            alpha = max(alpha, 0.15) * 0.7

            y0, y1 = int(i * cell_px), int((i + 1) * cell_px)
            x0, x1 = int(j * cell_px), int((j + 1) * cell_px)
            y0, y1 = max(0, y0), min(S, y1)
            x0, x1 = max(0, x0), min(S, x1)

            bg = pixels[y0:y1, x0:x1].astype(np.float32)
            pixels[y0:y1, x0:x1] = (bg * (1 - alpha) + mixed * alpha).astype(np.uint8)

    return pixels


# ── 3D Occupancy → BEV collapse ────────────────────────────────────────────

def render_occupancy_bev(voxel_grid, mode='argmax', target_class=0,
                         grid_range=51.2, resolution=0.512,
                         class_names=None, class_colors=None):
    """Render a 3D occupancy grid as a BEV image by collapsing the Z axis.

    Args:
        voxel_grid: [C, D, H, W] numpy array of per-class occupancy logits,
                    where D is the depth (height) dimension.
                    Can also be [C, H, W] (already 2D) — delegates to render_bev.
        mode:       'argmax' | 'class_heatmap' | 'composite'
        target_class: int for class_heatmap mode.
        grid_range: half-range in metres.
        resolution: cell size in metres.
        class_names: list of class name strings (optional).
        class_colors: dict of name→(r,g,b) (optional).

    Returns:
        PIL.Image (RGB) 800×800.
    """
    if voxel_grid is None:
        return render_bev(None, mode=mode, target_class=target_class,
                          grid_range=grid_range, resolution=resolution,
                          class_names=class_names, class_colors=class_colors)

    if voxel_grid.ndim == 3:
        # Already 2D BEV — just delegate
        return render_bev(voxel_grid, mode=mode, target_class=target_class,
                          grid_range=grid_range, resolution=resolution,
                          class_names=class_names, class_colors=class_colors)

    if voxel_grid.ndim != 4:
        raise ValueError(f"Expected [C,D,H,W] or [C,H,W], got shape {voxel_grid.shape}")

    # Collapse Z (depth) axis: max-pool per class → [C, H, W]
    bev_grid = np.max(voxel_grid, axis=1)  # [C, D, H, W] → [C, H, W]

    return render_bev(bev_grid, mode=mode, target_class=target_class,
                      grid_range=grid_range, resolution=resolution,
                      class_names=class_names, class_colors=class_colors)
