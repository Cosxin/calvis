"""Camera panel rendering with heatmap overlay."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _get_font(size=13):
    """Try to load a monospace font, fall back to default."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("Menlo.ttc", size)
        except (OSError, IOError):
            return ImageFont.load_default()


def _colormap_turbo(val):
    """Simple turbo-like colormap: 0->blue, 0.25->cyan, 0.5->green, 0.75->yellow, 1->red."""
    val = float(np.clip(val, 0.0, 1.0))
    if val < 0.25:
        t = val / 0.25
        r, g, b = 0, int(t * 255), 255
    elif val < 0.5:
        t = (val - 0.25) / 0.25
        r, g, b = 0, 255, int((1 - t) * 255)
    elif val < 0.75:
        t = (val - 0.5) / 0.25
        r, g, b = int(t * 255), 255, 0
    else:
        t = (val - 0.75) / 0.25
        r, g, b = 255, int((1 - t) * 255), 0
    return r, g, b


def render_camera(image, heatmap=None, camera_name='',
                  projection_point=None, color='#00ccff'):
    """Render camera view with optional heatmap overlay.

    Args:
        image: PIL Image (original camera image).
        heatmap: [H, W] numpy array in [0, 1], overlaid as semi-transparent
                 colormap. Can be None.
        camera_name: string drawn in top-left corner.
        projection_point: (u, v) pixel coords to draw crosshair marker, or None.
        color: hex color string for this camera's accent color.

    Returns:
        PIL Image at camera resolution.
    """
    if image is None:
        # Create a placeholder dark image
        img = Image.new('RGB', (800, 450), color=(13, 13, 13))
        draw = ImageDraw.Draw(img)
        font = _get_font(14)
        label = camera_name or "No Image"
        draw.text((12, 12), label, fill=(100, 100, 100), font=font)
        return img

    # Work on a copy
    img = image.copy().convert('RGB')
    w, h = img.size

    # --- Apply heatmap overlay ---
    if heatmap is not None:
        hm = np.array(heatmap, dtype=np.float32)
        # Resize heatmap to image dimensions if needed
        if hm.shape[0] != h or hm.shape[1] != w:
            from PIL import Image as PILImage
            hm_img = PILImage.fromarray((hm * 255).astype(np.uint8), mode='L')
            hm_img = hm_img.resize((w, h), PILImage.BILINEAR)
            hm = np.array(hm_img).astype(np.float32) / 255.0

        # Build colormap overlay
        overlay_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        overlay_alpha = np.zeros((h, w), dtype=np.float32)

        # Vectorized colormap application
        hm_clipped = np.clip(hm, 0.0, 1.0)

        # Turbo-like colormap vectorized
        r = np.zeros_like(hm_clipped)
        g = np.zeros_like(hm_clipped)
        b = np.zeros_like(hm_clipped)

        # Region 0 - 0.25: blue to cyan
        mask = hm_clipped < 0.25
        t = hm_clipped[mask] / 0.25
        r[mask] = 0
        g[mask] = t
        b[mask] = 1.0

        # Region 0.25 - 0.5: cyan to green
        mask = (hm_clipped >= 0.25) & (hm_clipped < 0.5)
        t = (hm_clipped[mask] - 0.25) / 0.25
        r[mask] = 0
        g[mask] = 1.0
        b[mask] = 1.0 - t

        # Region 0.5 - 0.75: green to yellow
        mask = (hm_clipped >= 0.5) & (hm_clipped < 0.75)
        t = (hm_clipped[mask] - 0.5) / 0.25
        r[mask] = t
        g[mask] = 1.0
        b[mask] = 0

        # Region 0.75 - 1.0: yellow to red
        mask = hm_clipped >= 0.75
        t = (hm_clipped[mask] - 0.75) / 0.25
        r[mask] = 1.0
        g[mask] = 1.0 - t
        b[mask] = 0

        overlay_rgb[:, :, 0] = (r * 255).astype(np.uint8)
        overlay_rgb[:, :, 1] = (g * 255).astype(np.uint8)
        overlay_rgb[:, :, 2] = (b * 255).astype(np.uint8)

        # Alpha proportional to heatmap intensity (stronger = more visible)
        overlay_alpha = hm_clipped * 0.6  # max 60% opacity

        # Blend
        img_arr = np.array(img).astype(np.float32)
        alpha_3d = overlay_alpha[:, :, np.newaxis]
        blended = img_arr * (1 - alpha_3d) + overlay_rgb.astype(np.float32) * alpha_3d
        img = Image.fromarray(blended.astype(np.uint8))

    draw = ImageDraw.Draw(img)

    # Parse hex color
    color_hex = color.lstrip('#')
    accent_rgb = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))

    # --- Draw camera name label ---
    font = _get_font(14)
    small_font = _get_font(11)
    if camera_name:
        # Background rectangle for label
        text_w = len(camera_name) * 9 + 16
        draw.rectangle([0, 0, text_w, 24], fill=(0, 0, 0, 180))
        draw.text((8, 5), camera_name.upper(), fill=accent_rgb, font=font)

    # --- Draw projection crosshair ---
    if projection_point is not None:
        u, v = projection_point
        u, v = int(u), int(v)
        if 0 <= u < w and 0 <= v < h:
            arm_len = 25
            # Crosshair lines
            draw.line([(u - arm_len, v), (u + arm_len, v)], fill=accent_rgb, width=2)
            draw.line([(u, v - arm_len), (u, v + arm_len)], fill=accent_rgb, width=2)
            # Center dot
            dot_r = 5
            draw.ellipse([u - dot_r, v - dot_r, u + dot_r, v + dot_r], fill=accent_rgb)
            # Outer circle
            circle_r = 16
            draw.ellipse(
                [u - circle_r, v - circle_r, u + circle_r, v + circle_r],
                outline=accent_rgb, width=2
            )
            # Coordinate label
            coord_text = f"({u}, {v})"
            draw.text((u + 20, v - 8), coord_text, fill=accent_rgb, font=small_font)

    # --- Draw border in camera color ---
    draw.rectangle([0, 0, w - 1, h - 1], outline=accent_rgb, width=2)

    return img
