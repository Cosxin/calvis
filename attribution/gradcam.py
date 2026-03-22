"""
GradCAM attribution for BEV models using Captum.
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _find_last_conv_layer(model) -> str:
    """Auto-detect the last convolutional layer in the model's backbone.

    Walks the model's named modules in reverse order and returns the name
    of the last Conv2d layer found. Prefers layers inside 'backbone' or
    'img_backbone' submodules.

    Args:
        model: PyTorch model.

    Returns:
        Dotted name string for the layer (e.g., 'backbone.layer4.2.conv3').

    Raises:
        RuntimeError: If no Conv2d layer is found.
    """
    last_conv_name = None
    last_backbone_conv_name = None

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv_name = name
            # Prefer layers inside backbone-like submodules
            lower = name.lower()
            if any(kw in lower for kw in ('backbone', 'img_backbone', 'encoder')):
                last_backbone_conv_name = name

    result = last_backbone_conv_name or last_conv_name
    if result is None:
        raise RuntimeError(
            "Could not find any Conv2d layer in the model. "
            "Please specify layer_name explicitly."
        )

    logger.info("Auto-detected last conv layer: %s", result)
    return result


def _get_module_by_name(model, layer_name: str) -> torch.nn.Module:
    """Retrieve a submodule by its dotted name string.

    Args:
        model: The parent model.
        layer_name: Dotted path like 'backbone.layer4.2.conv3'.

    Returns:
        The target module.

    Raises:
        AttributeError: If the path is invalid.
    """
    parts = layer_name.split('.')
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def _build_captum_forward(model, sample: dict, cell_i: int, cell_j: int,
                          class_idx: int, device: str):
    """Build a Captum-compatible forward function for GradCAM.

    Returns a callable that takes [1, 6, 3, H, W] and returns a scalar.
    """
    from pipeline.wrapper import forward_fn

    def captum_forward(images_batch: torch.Tensor) -> torch.Tensor:
        images = images_batch.squeeze(0)  # [6, 3, H, W]
        modified_sample = dict(sample)
        modified_sample['image_tensors'] = images
        scalar = forward_fn(model, modified_sample, cell_i, cell_j,
                            class_idx=class_idx, device=device)
        # Captum expects output[0] to be indexable — wrap scalar as [1] tensor
        return scalar.unsqueeze(0)

    return captum_forward


def attr_gradcam(model, sample: dict, cell_i: int, cell_j: int,
                 class_idx: int = 0, layer_name: str = None,
                 device: str = 'cpu') -> np.ndarray:
    """Compute GradCAM attribution.

    Uses Captum's LayerGradCam when possible. Falls back to a manual
    hook-based implementation if Captum's version encounters issues.

    Args:
        model: The loaded BEV model.
        sample: Sample dict containing 'image_tensors' [6, 3, H, W].
        cell_i: BEV grid row index.
        cell_j: BEV grid column index.
        class_idx: Class channel index in the BEV output.
        layer_name: Dotted name of the target conv layer. If None, auto-detects
                     the last conv layer in the backbone.
        device: Torch device string.

    Returns:
        [6, H, W] numpy array of attribution heatmaps, normalized to [0, 1].
    """
    from .utils import postprocess_heatmap

    try:
        from captum.attr import LayerGradCam
    except ImportError as e:
        raise ImportError(
            "Captum is required for GradCAM. Install it with: pip install captum"
        ) from e

    try:
        # Auto-detect layer if not specified
        if layer_name is None:
            layer_name = _find_last_conv_layer(model)

        target_layer = _get_module_by_name(model, layer_name)

        images_tensor = sample['image_tensors']
        if not isinstance(images_tensor, torch.Tensor):
            images_tensor = torch.tensor(images_tensor)
        images_tensor = images_tensor.to(device).float()
        images_tensor.requires_grad_(True)

        input_tensor = images_tensor.unsqueeze(0)  # [1, 6, 3, H, W]

        captum_fwd = _build_captum_forward(
            model, sample, cell_i, cell_j, class_idx, device
        )

        gradcam = LayerGradCam(captum_fwd, target_layer)
        attributions = gradcam.attribute(input_tensor, target=None)
        # attributions shape depends on layer output shape
        # Typically [1, C_layer, H_feat, W_feat] or similar

        attr = attributions.squeeze(0).detach().cpu().numpy()

        # The layer output might have shape [6*C, H_feat, W_feat] or
        # [C, H_feat, W_feat]. We need to reshape to [6, H_cam, W_cam].
        n_cams = 6
        img_h, img_w = images_tensor.shape[-2:]

        if attr.ndim == 3:
            # If the attribution has more channels than 6, average groups
            n_channels = attr.shape[0]
            if n_channels >= n_cams and n_channels % n_cams == 0:
                # Reshape to [6, C_per_cam, H_feat, W_feat]
                c_per_cam = n_channels // n_cams
                attr = attr.reshape(n_cams, c_per_cam, attr.shape[1], attr.shape[2])
                attr = np.mean(np.abs(attr), axis=1)  # [6, H_feat, W_feat]
            else:
                # Average across channels and replicate for all cameras
                attr_2d = np.mean(np.abs(attr), axis=0)  # [H_feat, W_feat]
                attr = np.stack([attr_2d] * n_cams, axis=0)  # [6, H_feat, W_feat]
        elif attr.ndim == 2:
            attr = np.abs(attr)
            attr = np.stack([attr] * n_cams, axis=0)
        elif attr.ndim == 4:
            # [6, C, H_feat, W_feat] - average over channels
            attr = np.mean(np.abs(attr), axis=1)  # [6, H_feat, W_feat]
        else:
            logger.warning("Unexpected attribution ndim=%d, flattening.", attr.ndim)
            attr = np.abs(attr).reshape(n_cams, -1)
            side = int(np.sqrt(attr.shape[1]))
            if side * side == attr.shape[1]:
                attr = attr.reshape(n_cams, side, side)
            else:
                attr = np.zeros((n_cams, img_h, img_w))

        # Upsample feature-level heatmaps to image resolution
        feat_h, feat_w = attr.shape[-2:]
        if feat_h != img_h or feat_w != img_w:
            attr_tensor = torch.tensor(attr, dtype=torch.float32).unsqueeze(1)
            # [6, 1, H_feat, W_feat]
            attr_upsampled = F.interpolate(
                attr_tensor, size=(img_h, img_w),
                mode='bilinear', align_corners=False
            )
            attr = attr_upsampled.squeeze(1).numpy()  # [6, H, W]

        # Apply ReLU (GradCAM convention: keep only positive attributions)
        attr = np.maximum(attr, 0)

        result = postprocess_heatmap(attr, sigma=3.0)

        logger.info(
            "GradCAM attribution computed: shape=%s, layer='%s'",
            result.shape, layer_name,
        )
        return result

    except Exception as e:
        logger.error("GradCAM failed: %s", str(e), exc_info=True)
        # Fallback: try manual gradient * activation approach
        return _gradcam_manual_fallback(
            model, sample, cell_i, cell_j, class_idx, layer_name, device
        )


def _gradcam_manual_fallback(model, sample: dict, cell_i: int, cell_j: int,
                             class_idx: int, layer_name: str,
                             device: str) -> np.ndarray:
    """Manual GradCAM fallback using forward/backward hooks.

    Used when Captum's LayerGradCam encounters issues (e.g., with complex
    model architectures).
    """
    from .utils import postprocess_heatmap
    from pipeline.wrapper import forward_fn

    try:
        if layer_name is None:
            layer_name = _find_last_conv_layer(model)

        target_layer = _get_module_by_name(model, layer_name)

        activations = {}
        gradients = {}

        def fwd_hook(module, inp, out):
            activations['value'] = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            gradients['value'] = grad_out[0].detach()

        fwd_handle = target_layer.register_forward_hook(fwd_hook)
        bwd_handle = target_layer.register_full_backward_hook(bwd_hook)

        try:
            images_tensor = sample['image_tensors']
            if not isinstance(images_tensor, torch.Tensor):
                images_tensor = torch.tensor(images_tensor)
            images_tensor = images_tensor.to(device).float().requires_grad_(True)

            modified_sample = dict(sample)
            modified_sample['image_tensors'] = images_tensor

            output = forward_fn(model, modified_sample, cell_i, cell_j,
                                class_idx=class_idx, device=device)
            output.backward()

            if 'value' not in activations or 'value' not in gradients:
                raise RuntimeError("Hooks did not capture activations/gradients.")

            act = activations['value'].cpu().numpy()  # [batch?, C, H, W]
            grad = gradients['value'].cpu().numpy()

            # Global average pooling of gradients to get channel weights
            if grad.ndim == 4:
                weights = np.mean(grad, axis=(2, 3), keepdims=True)
            elif grad.ndim == 3:
                weights = np.mean(grad, axis=2, keepdims=True)
            else:
                weights = grad

            # Weighted combination of activations
            cam = np.sum(weights * act, axis=-3 if act.ndim >= 3 else 0)
            cam = np.maximum(cam, 0)  # ReLU

            # Reshape to [6, H_cam, W_cam]
            n_cams = 6
            img_h, img_w = sample['image_tensors'].shape[-2:]

            if cam.ndim == 2:
                cam = np.stack([cam] * n_cams, axis=0)
            elif cam.ndim == 1:
                side = int(np.sqrt(cam.shape[0]))
                cam = cam.reshape(side, side)
                cam = np.stack([cam] * n_cams, axis=0)

            # Upsample to image resolution
            if cam.shape[-2:] != (img_h, img_w):
                import torch.nn.functional as F
                cam_t = torch.tensor(cam, dtype=torch.float32).unsqueeze(1)
                cam_t = F.interpolate(cam_t, size=(img_h, img_w),
                                      mode='bilinear', align_corners=False)
                cam = cam_t.squeeze(1).numpy()

            result = postprocess_heatmap(cam, sigma=3.0)
            logger.info("GradCAM (manual fallback) computed: shape=%s", result.shape)
            return result

        finally:
            fwd_handle.remove()
            bwd_handle.remove()

    except Exception as e:
        logger.error("GradCAM manual fallback also failed: %s", str(e), exc_info=True)
        h, w = sample['image_tensors'].shape[-2:]
        logger.warning(
            "Returning zero heatmaps of shape [6, %d, %d] due to GradCAM failure.",
            h, w,
        )
        return np.zeros((6, h, w), dtype=np.float64)
