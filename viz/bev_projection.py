"""Project camera-space attention heatmaps into BEV grid coordinates."""

import numpy as np
from PIL import Image, ImageDraw


def project_heatmap_to_bev(heatmap, K, E, grid_range=51.2, resolution=0.512, grid_cells=200):
    """Vectorized projection of a camera attention heatmap into BEV space.

    For each BEV cell, projects the ground-plane point to the camera pixel
    and samples the heatmap value. Cells outside the camera FOV remain 0.

    Args:
        heatmap: [H, W] numpy array in [0, 1] — attention in camera space.
        K: [3, 3] camera intrinsic matrix.
        E: [4, 4] ego-to-camera extrinsic matrix.
        grid_range: half-range in meters (grid spans -grid_range to +grid_range).
        resolution: meters per cell.
        grid_cells: number of cells per side.

    Returns:
        [grid_cells, grid_cells] numpy array with projected attention values.
    """
    h_img, w_img = heatmap.shape

    # Build BEV cell center coordinates
    js, is_ = np.meshgrid(np.arange(grid_cells), np.arange(grid_cells))
    wx = -grid_range + (js + 0.5) * resolution   # right in BEV world
    wz = grid_range - (is_ + 0.5) * resolution   # forward in BEV world

    # BEV world → ego frame: ego_x = wz (forward), ego_y = -wx (left), ego_z = 0
    ego_x = wz.ravel()
    ego_y = -wx.ravel()
    ego_z = np.zeros_like(ego_x)
    ones = np.ones_like(ego_x)
    ego_pts = np.stack([ego_x, ego_y, ego_z, ones], axis=1)  # [N, 4]

    # Ego → camera frame
    cam_pts = (E @ ego_pts.T).T  # [N, 4]

    # Depth check: z > 0 (in front of camera)
    depth = cam_pts[:, 2]
    valid = depth > 0.1

    # Camera → pixel
    cam_xyz = cam_pts[:, :3]
    px = (K @ cam_xyz.T).T
    u = np.where(valid, px[:, 0] / np.maximum(depth, 0.1), -1)
    v = np.where(valid, px[:, 1] / np.maximum(depth, 0.1), -1)

    # Bounds check
    u_int = u.astype(np.int32)
    v_int = v.astype(np.int32)
    in_bounds = valid & (u_int >= 0) & (u_int < w_img) & (v_int >= 0) & (v_int < h_img)

    # Sample heatmap
    bev_flat = np.zeros(grid_cells * grid_cells, dtype=np.float32)
    valid_idx = np.where(in_bounds)[0]
    bev_flat[valid_idx] = heatmap[v_int[valid_idx], u_int[valid_idx]]

    return bev_flat.reshape(grid_cells, grid_cells)


def render_vlm_bev(vlm_bev_attn, lss_bev_mask=None, gt_boxes=None,
                   grid_range=51.2, resolution=0.512, image_size=800):
    """Render front-camera BEV view: ego at bottom-center, forward is up.

    Crops to the forward half of the grid (ego → 51.2m ahead) and the
    horizontal extent of the front camera FOV. Renders VLM attention as
    turbo colormap and LSS detections as cyan overlays.

    Args:
        vlm_bev_attn: [grid_cells, grid_cells] attention values in [0, 1].
        lss_bev_mask: [grid_cells, grid_cells] bool mask of LSS detections, or None.
        grid_range: BEV half-range in meters.
        resolution: meters per cell.
        image_size: output image height in pixels.

    Returns:
        PIL Image (RGB).
    """
    grid_cells = vlm_bev_attn.shape[0]
    half = grid_cells // 2

    # Crop to forward half (rows 0..half = forward region in BEV)
    # and horizontal center ± some margin for front camera FOV
    # Front camera typically covers ~60° → at 50m, lateral extent ~±29m → ~57 cells
    # Use generous crop: center ± 70 cells (±35.8m)
    lateral_half = min(70, half)
    col_start = half - lateral_half
    col_end = half + lateral_half

    # Crop both attention and LSS mask
    attn_crop = vlm_bev_attn[0:half, col_start:col_end]
    lss_crop = lss_bev_mask[0:half, col_start:col_end] if lss_bev_mask is not None else None

    crop_h, crop_w = attn_crop.shape
    # Output image: maintain aspect ratio
    aspect = crop_w / crop_h
    img_w = int(image_size * aspect)
    img_h = image_size

    # Dark background
    img = Image.new('RGB', (img_w, img_h), color=(10, 10, 10))
    arr = np.array(img, dtype=np.float32)

    # Turbo colormap for VLM attention
    attn = np.clip(attn_crop, 0, 1)
    mask = attn > 0.01

    if mask.any():
        attn_img = Image.fromarray((attn * 255).astype(np.uint8), mode='L')
        attn_img = attn_img.resize((img_w, img_h), Image.BILINEAR)
        attn_up = np.array(attn_img).astype(np.float32) / 255.0

        # Turbo colormap (vectorized)
        r = np.zeros_like(attn_up)
        g = np.zeros_like(attn_up)
        b = np.zeros_like(attn_up)

        m1 = attn_up < 0.25
        t = attn_up[m1] / 0.25
        r[m1], g[m1], b[m1] = 0, t, 1.0

        m2 = (attn_up >= 0.25) & (attn_up < 0.5)
        t = (attn_up[m2] - 0.25) / 0.25
        r[m2], g[m2], b[m2] = 0, 1.0, 1.0 - t

        m3 = (attn_up >= 0.5) & (attn_up < 0.75)
        t = (attn_up[m3] - 0.5) / 0.25
        r[m3], g[m3], b[m3] = t, 1.0, 0

        m4 = attn_up >= 0.75
        t = (attn_up[m4] - 0.75) / 0.25
        r[m4], g[m4], b[m4] = 1.0, 1.0 - t, 0

        overlay = np.stack([r * 255, g * 255, b * 255], axis=-1)

        mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode='L')
        mask_img = mask_img.resize((img_w, img_h), Image.NEAREST)
        mask_up = np.array(mask_img).astype(np.float32) / 255.0

        alpha = attn_up * mask_up * 0.7
        alpha_3d = alpha[:, :, np.newaxis]
        arr = arr * (1 - alpha_3d) + overlay * alpha_3d

    img = Image.fromarray(arr.astype(np.uint8))
    draw = ImageDraw.Draw(img)

    # Draw LSS detections as cyan overlay
    if lss_crop is not None and lss_crop.any():
        lss_up = Image.fromarray((lss_crop.astype(np.uint8) * 255), mode='L')
        lss_up = lss_up.resize((img_w, img_h), Image.NEAREST)
        lss_arr = np.array(lss_up) > 128

        img_arr = np.array(img)
        from scipy.ndimage import binary_dilation
        dilated = binary_dilation(lss_arr, iterations=2)
        border = dilated & ~lss_arr
        img_arr[border] = [0, 220, 255]
        img_arr[lss_arr] = (img_arr[lss_arr] * 0.6 + np.array([0, 180, 220]) * 0.4).astype(np.uint8)
        img = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(img)

    # Draw GT bounding boxes (outline only, per-class color)
    if gt_boxes:
        from viz.bev import CLASS_COLORS_BY_IDX
        for box in gt_boxes:
            corners = box.get('corners')
            if not corners or len(corners) < 4:
                continue
            cls_idx = box.get('class_idx', 0)
            color = CLASS_COLORS_BY_IDX[cls_idx] if cls_idx < len(CLASS_COLORS_BY_IDX) else (200, 200, 200)
            # Transform ego-frame corners (x=forward, y=-left) to cropped BEV pixels
            # In the full BEV grid: col = (ego_y_neg + grid_range) / (2*grid_range) * grid_cells
            #   where ego_y_neg = -ego_y (BEV x-axis = right = -ego_y)
            # row = (grid_range - ego_x) / (2*grid_range) * grid_cells
            #   where ego_x = forward
            # Then crop: row stays (forward half = rows 0..half), col -= col_start
            pts = []
            for bev_wx, bev_wz in corners:
                # Corners already in BEV world coords: wx=right, wz=forward
                # Full BEV grid: col = (wx + grid_range) / (2*grid_range) * grid_cells
                #                row = (grid_range - wz) / (2*grid_range) * grid_cells
                full_col = (bev_wx + grid_range) / (2 * grid_range) * grid_cells
                full_row = (grid_range - bev_wz) / (2 * grid_range) * grid_cells
                # Crop coords
                crop_col = full_col - col_start
                crop_row = full_row  # forward half = rows 0..half
                # To image pixels
                px_x = crop_col / crop_w * img_w
                px_y = crop_row / crop_h * img_h
                pts.append((px_x, px_y))
            # Only draw if at least some of the box is within the view
            if any(0 <= p[0] <= img_w and 0 <= p[1] <= img_h for p in pts):
                pts.append(pts[0])  # close the polygon
                draw.line(pts, fill=color, width=2)

    # Draw ego marker at bottom center
    cx = img_w // 2
    cy = img_h - 12
    draw.polygon([(cx, cy - 10), (cx - 6, cy + 4), (cx + 6, cy + 4)],
                 fill=(255, 255, 255), outline=(200, 200, 200))
    draw.text((cx + 10, cy - 8), "ego", fill=(200, 200, 200))

    # Distance markers (horizontal dashed lines every 10m)
    meters_per_row = resolution
    try:
        font = None
        for dist_m in [10, 20, 30, 40, 50]:
            # Row in cropped grid: dist_m maps to (half - dist_m/resolution) in full grid
            # In cropped grid (flipped): row = half - dist_m/resolution
            row_in_crop = crop_h - int(dist_m / resolution)
            if row_in_crop < 0 or row_in_crop >= crop_h:
                continue
            y_px = int(row_in_crop / crop_h * img_h)
            # Dashed line
            for x in range(0, img_w, 12):
                draw.line([(x, y_px), (min(x + 6, img_w), y_px)], fill=(50, 70, 70), width=1)
            draw.text((4, y_px - 10), f"{dist_m}m", fill=(80, 100, 100))
    except Exception:
        pass

    # Border
    draw.rectangle([0, 0, img_w - 1, img_h - 1], outline=(40, 60, 60), width=1)

    # "Forward ↑" label at top
    draw.text((img_w // 2 - 20, 4), "forward", fill=(80, 100, 100))

    return img
