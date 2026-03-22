"""
Unified attribution interface and heatmap post-processing utilities.
"""

import logging
import numpy as np
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


def normalize_heatmap(heatmap: np.ndarray, method: str = 'percentile',
                      percentile: float = 99) -> np.ndarray:
    """Normalize heatmap to 0-1 range.

    Args:
        heatmap: Array of any shape with raw attribution values.
        method: 'percentile' clips at the given percentile then scales,
                'minmax' uses simple min-max normalization,
                'abs' takes absolute values then applies percentile normalization.
        percentile: Upper percentile for clipping (used with 'percentile' and 'abs').

    Returns:
        Array of same shape with values in [0, 1].
    """
    heatmap = np.array(heatmap, dtype=np.float64)

    if method == 'abs':
        heatmap = np.abs(heatmap)
        method = 'percentile'

    if method == 'percentile':
        vmin = 0.0
        vmax = np.percentile(heatmap, percentile) if heatmap.size > 0 else 1.0
        if vmax <= vmin:
            vmax = np.max(heatmap)
        if vmax <= vmin:
            return np.zeros_like(heatmap, dtype=np.float64)
        heatmap = np.clip(heatmap, vmin, vmax)
        heatmap = (heatmap - vmin) / (vmax - vmin)
    elif method == 'minmax':
        vmin = np.min(heatmap)
        vmax = np.max(heatmap)
        if vmax - vmin < 1e-10:
            return np.zeros_like(heatmap, dtype=np.float64)
        heatmap = (heatmap - vmin) / (vmax - vmin)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return np.clip(heatmap, 0.0, 1.0).astype(np.float64)


def smooth_heatmap(heatmap: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Apply Gaussian smoothing to a heatmap.

    If heatmap has shape [N, H, W], each slice is smoothed independently.
    If heatmap has shape [H, W], it is smoothed directly.

    Args:
        heatmap: Heatmap array, either [H, W] or [N, H, W].
        sigma: Standard deviation for Gaussian kernel.

    Returns:
        Smoothed heatmap of the same shape.
    """
    if sigma <= 0:
        return heatmap

    heatmap = np.array(heatmap, dtype=np.float64)

    if heatmap.ndim == 3:
        for i in range(heatmap.shape[0]):
            heatmap[i] = gaussian_filter(heatmap[i], sigma=sigma)
    elif heatmap.ndim == 2:
        heatmap = gaussian_filter(heatmap, sigma=sigma)
    else:
        logger.warning("smooth_heatmap: unexpected ndim=%d, skipping", heatmap.ndim)

    return heatmap


def postprocess_heatmap(heatmap: np.ndarray, sigma: float = 3.0,
                        norm_method: str = 'percentile',
                        percentile: float = 99,
                        per_camera: bool = False) -> np.ndarray:
    """Standard post-processing pipeline: abs -> smooth -> normalize.

    Args:
        heatmap: Raw attribution array, typically [6, H, W].
        sigma: Gaussian smoothing sigma. Set to 0 to skip.
        norm_method: Normalization method for normalize_heatmap.
        percentile: Percentile for clipping.
        per_camera: If True, normalize each camera independently. This
            prevents a noisy camera from dominating the visualization.

    Returns:
        Post-processed heatmap in [0, 1].
    """
    heatmap = np.abs(heatmap).astype(np.float64)
    if sigma > 0:
        heatmap = smooth_heatmap(heatmap, sigma=sigma)

    if per_camera and heatmap.ndim == 3:
        # Log per-camera magnitudes for debugging
        for cam_i in range(heatmap.shape[0]):
            mag = float(np.sum(heatmap[cam_i]))
            if mag > 0:
                logger.debug("  cam %d attribution magnitude: %.4f", cam_i, mag)
        # Normalize each camera slice independently
        for cam_i in range(heatmap.shape[0]):
            heatmap[cam_i] = normalize_heatmap(
                heatmap[cam_i], method=norm_method, percentile=percentile
            )
    else:
        heatmap = normalize_heatmap(heatmap, method=norm_method, percentile=percentile)
    return heatmap


def attribute(model, sample: dict, cell_i: int, cell_j: int,
              class_idx: int = 0, method: str = 'gradcam',
              device: str = 'cpu', **kwargs) -> np.ndarray:
    """Unified attribution interface.

    Dispatches to the appropriate attribution method and returns a
    [6, H, W] numpy array normalized to [0, 1].

    Args:
        model: The loaded BEV model.
        sample: Sample dict with 'image_tensors', 'images', etc.
        cell_i: BEV grid row index.
        cell_j: BEV grid column index.
        class_idx: Class channel index in the BEV output.
        method: One of 'ig', 'gradcam', 'attention', 'occlusion'.
        device: Torch device string.
        **kwargs: Extra arguments forwarded to the specific method.

    Returns:
        [6, H, W] numpy array with attribution heatmaps in [0, 1].

    Raises:
        ValueError: If method is unknown.
    """
    method = method.lower().strip()

    if method == 'ig':
        from .integrated_gradients import attr_ig
        return attr_ig(model, sample, cell_i, cell_j,
                       class_idx=class_idx, device=device, **kwargs)
    elif method == 'gradcam':
        from .gradcam import attr_gradcam
        return attr_gradcam(model, sample, cell_i, cell_j,
                            class_idx=class_idx, device=device, **kwargs)
    elif method == 'attention':
        from .attention import attr_attention
        return attr_attention(model, sample, cell_i, cell_j,
                              device=device, **kwargs)
    elif method == 'occlusion':
        from .occlusion import attr_occlusion
        return attr_occlusion(model, sample, cell_i, cell_j,
                              class_idx=class_idx, device=device, **kwargs)
    else:
        raise ValueError(
            f"Unknown attribution method '{method}'. "
            f"Choose from: 'ig', 'gradcam', 'attention', 'occlusion'."
        )
