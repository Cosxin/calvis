"""
Attention-based attribution for BEV models.

Extracts cross-attention weights from transformer layers when available
(e.g., BEVFormer). Falls back to gradient * input if the model does not
expose attention maps.
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _find_attention_layers(model):
    """Search for transformer attention layers in the model.

    Looks for modules that are likely cross-attention layers connecting
    image features to BEV queries (common in BEVFormer-style models).

    Returns:
        List of (name, module) tuples for attention layers, or empty list.
    """
    attention_layers = []

    # Common attention module class names in BEV transformers
    attn_keywords = (
        'crossattn', 'cross_attn', 'cross_attention',
        'spatialcrossattention', 'spatial_cross_attention',
        'deformableattn', 'deformable_attn',
        'multiheadattention', 'multi_head_attention',
        'temporalselfatten', 'spatialatten',
    )

    for name, module in model.named_modules():
        module_type = type(module).__name__.lower()
        name_lower = name.lower()

        # Check if this looks like an attention layer
        is_attn = any(kw in module_type for kw in attn_keywords)
        is_attn = is_attn or any(kw in name_lower for kw in attn_keywords)

        if is_attn:
            attention_layers.append((name, module))

    return attention_layers


def _extract_attention_weights(model, sample: dict, cell_i: int, cell_j: int,
                               device: str) -> np.ndarray:
    """Extract cross-attention weights by hooking into attention layers.

    Registers forward hooks on attention layers to capture the attention
    weight matrices during a forward pass.

    Args:
        model: The BEV model with transformer attention layers.
        sample: Sample dict.
        cell_i: BEV grid row index.
        cell_j: BEV grid column index.
        device: Torch device string.

    Returns:
        [6, H, W] numpy array of attention-based heatmaps, or None if extraction fails.
    """
    from pipeline.wrapper import forward_fn

    attn_layers = _find_attention_layers(model)
    if not attn_layers:
        return None

    logger.info("Found %d attention layers: %s",
                len(attn_layers), [n for n, _ in attn_layers])

    captured_weights = []
    hooks = []

    def make_hook(layer_name):
        def hook_fn(module, inp, out):
            # Attention modules may return (output, weights) or just output.
            # We try to capture any attention weight tensors.
            weights = None

            if isinstance(out, tuple) and len(out) >= 2:
                # Second element is often attention weights
                candidate = out[1]
                if isinstance(candidate, torch.Tensor) and candidate.ndim >= 2:
                    weights = candidate.detach().cpu()

            # Some modules store attn_weights as an attribute after forward
            if weights is None:
                for attr_name in ('attn_weights', 'attention_weights',
                                  'attn_output_weights', '_attn_weights'):
                    w = getattr(module, attr_name, None)
                    if isinstance(w, torch.Tensor):
                        weights = w.detach().cpu()
                        break

            if weights is not None:
                captured_weights.append({
                    'name': layer_name,
                    'weights': weights.numpy(),
                })

        return hook_fn

    try:
        for name, module in attn_layers:
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

        # Run a forward pass to trigger the hooks
        images_tensor = sample['image_tensors']
        if not isinstance(images_tensor, torch.Tensor):
            images_tensor = torch.tensor(images_tensor)
        images_tensor = images_tensor.to(device).float()

        modified_sample = dict(sample)
        modified_sample['image_tensors'] = images_tensor

        with torch.no_grad():
            forward_fn(model, modified_sample, cell_i, cell_j,
                       class_idx=0, device=device)

    finally:
        for h in hooks:
            h.remove()

    if not captured_weights:
        logger.info("Attention hooks registered but no weights captured.")
        return None

    logger.info("Captured attention weights from %d layers.", len(captured_weights))

    # Process captured attention weights into per-camera heatmaps
    return _process_attention_weights(
        captured_weights, sample, cell_i, cell_j
    )


def _process_attention_weights(captured_weights: list, sample: dict,
                               cell_i: int, cell_j: int) -> np.ndarray:
    """Process raw attention weight captures into [6, H, W] heatmaps.

    BEV query at (cell_i, cell_j) attends to image feature positions.
    We aggregate attention weights across layers and heads, then map
    them back to image pixel space.

    Args:
        captured_weights: List of dicts with 'name' and 'weights' keys.
        sample: Sample dict.
        cell_i: BEV row index.
        cell_j: BEV col index.

    Returns:
        [6, H, W] numpy array, or None if processing fails.
    """
    n_cams = 6
    img_h, img_w = sample['image_tensors'].shape[-2:]

    aggregated = np.zeros((n_cams, img_h, img_w), dtype=np.float64)
    n_valid = 0

    for cap in captured_weights:
        weights = cap['weights']  # Various shapes depending on the layer

        try:
            # Attention weights are typically [batch, n_heads, n_queries, n_keys]
            # or [n_queries, n_keys] or [batch, n_queries, n_keys]
            if weights.ndim == 4:
                # Average over batch and heads
                w = np.mean(weights, axis=(0, 1))  # [n_queries, n_keys]
            elif weights.ndim == 3:
                w = np.mean(weights, axis=0)  # [n_queries, n_keys]
            elif weights.ndim == 2:
                w = weights
            else:
                continue

            n_queries, n_keys = w.shape

            # The BEV query index for cell (i, j) depends on the BEV grid layout.
            # Typically: query_idx = cell_i * W_bev + cell_j
            # We don't know W_bev exactly, so estimate from n_queries
            w_bev = int(np.sqrt(n_queries))
            if w_bev * w_bev != n_queries:
                # Not a square grid - try common BEV sizes
                for candidate in [200, 128, 100, 50]:
                    if n_queries == candidate * candidate:
                        w_bev = candidate
                        break
                else:
                    # Use the query closest to the proportion
                    w_bev = int(np.sqrt(n_queries))

            query_idx = cell_i * w_bev + cell_j
            if query_idx >= n_queries:
                query_idx = min(query_idx, n_queries - 1)

            # Get attention weights for this BEV query -> all keys
            attn_for_query = w[query_idx]  # [n_keys]

            # Keys correspond to flattened multi-camera feature maps
            # Distribute across cameras
            keys_per_cam = n_keys // n_cams if n_cams > 0 else n_keys
            remainder = n_keys - keys_per_cam * n_cams

            for cam_idx in range(n_cams):
                start = cam_idx * keys_per_cam
                end = start + keys_per_cam
                if cam_idx == n_cams - 1:
                    end += remainder  # give remainder to last camera

                cam_attn = attn_for_query[start:end]

                # Reshape to 2D feature map
                feat_size = int(np.sqrt(len(cam_attn)))
                if feat_size * feat_size == len(cam_attn):
                    cam_map = cam_attn.reshape(feat_size, feat_size)
                else:
                    # Find closest rectangular shape
                    h_feat = int(np.sqrt(len(cam_attn) * img_h / img_w))
                    w_feat = len(cam_attn) // max(h_feat, 1)
                    if h_feat * w_feat != len(cam_attn):
                        h_feat = 1
                        w_feat = len(cam_attn)
                    cam_map = cam_attn[:h_feat * w_feat].reshape(h_feat, w_feat)

                # Upsample to image resolution
                cam_tensor = torch.tensor(
                    cam_map, dtype=torch.float32
                ).unsqueeze(0).unsqueeze(0)
                cam_up = F.interpolate(
                    cam_tensor, size=(img_h, img_w),
                    mode='bilinear', align_corners=False
                )
                aggregated[cam_idx] += cam_up.squeeze().numpy()

            n_valid += 1

        except Exception as e:
            logger.debug("Could not process attention weights from '%s': %s",
                         cap['name'], e)
            continue

    if n_valid == 0:
        return None

    aggregated /= n_valid
    return aggregated


def _gradient_input_fallback(model, sample: dict, cell_i: int, cell_j: int,
                             device: str) -> np.ndarray:
    """Fallback: gradient * input attribution when attention is unavailable.

    Computes the element-wise product of input gradients and input values,
    which provides a simple but effective attribution signal.

    Returns:
        [6, H, W] numpy array normalized to [0, 1].
    """
    from .utils import postprocess_heatmap
    from pipeline.wrapper import forward_fn

    try:
        images_tensor = sample['image_tensors']
        if not isinstance(images_tensor, torch.Tensor):
            images_tensor = torch.tensor(images_tensor)
        images_tensor = images_tensor.to(device).float()
        images_tensor.requires_grad_(True)

        modified_sample = dict(sample)
        modified_sample['image_tensors'] = images_tensor

        output = forward_fn(model, modified_sample, cell_i, cell_j,
                            class_idx=0, device=device)
        output.backward()

        if images_tensor.grad is None:
            raise RuntimeError("No gradient computed for input images.")

        grad = images_tensor.grad.detach().cpu().numpy()  # [6, 3, H, W]
        inp = images_tensor.detach().cpu().numpy()  # [6, 3, H, W]

        # gradient * input, summed over channels
        attr = np.sum(np.abs(grad * inp), axis=1)  # [6, H, W]

        # Log per-camera magnitude to help debug camera mismatch
        for ci in range(min(6, attr.shape[0])):
            logger.info("  cam %d (%s) grad*input magnitude: %.6f",
                        ci, ['F','FR','BR','B','BL','FL'][ci] if ci < 6 else '?',
                        float(np.sum(attr[ci])))

        result = postprocess_heatmap(attr, sigma=3.0, per_camera=True)
        logger.info("Gradient*Input fallback computed: shape=%s", result.shape)
        return result

    except Exception as e:
        logger.error("Gradient*Input fallback failed: %s", str(e), exc_info=True)
        h, w = sample['image_tensors'].shape[-2:]
        return np.zeros((6, h, w), dtype=np.float64)


def attr_attention(model, sample: dict, cell_i: int, cell_j: int,
                   device: str = 'cpu') -> np.ndarray:
    """Extract cross-attention weights from transformer layers.

    If the model has transformer cross-attention layers (e.g., BEVFormer),
    hooks into those layers to extract and aggregate attention weights for
    the BEV query at (cell_i, cell_j).

    Falls back to gradient * input attribution if:
    - No attention layers are found in the model
    - Attention weight extraction fails
    - The captured weights cannot be processed

    Args:
        model: The loaded BEV model.
        sample: Sample dict containing 'image_tensors' [6, 3, H, W].
        cell_i: BEV grid row index.
        cell_j: BEV grid column index.
        device: Torch device string.

    Returns:
        [6, H, W] numpy array of attribution heatmaps, normalized to [0, 1].
    """
    from .utils import postprocess_heatmap

    try:
        # First, try attention extraction
        result = _extract_attention_weights(model, sample, cell_i, cell_j, device)

        if result is not None:
            result = postprocess_heatmap(result, sigma=3.0)
            logger.info("Attention attribution computed: shape=%s", result.shape)
            return result

        logger.info(
            "Attention extraction returned None, falling back to gradient*input."
        )

    except Exception as e:
        logger.warning("Attention extraction failed: %s. Using fallback.", str(e))

    # Fallback to gradient * input
    return _gradient_input_fallback(model, sample, cell_i, cell_j, device)
