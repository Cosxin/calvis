"""
Integrated Gradients attribution for BEV models using Captum.
"""

import logging
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


def _make_baseline(images_tensor: torch.Tensor, baseline_type: str,
                   device: str) -> torch.Tensor:
    """Create a baseline tensor for Integrated Gradients.

    Args:
        images_tensor: Input images tensor [6, 3, H, W].
        baseline_type: 'black' (zeros), 'blur' (gaussian blur), or 'noise' (random).
        device: Torch device string.

    Returns:
        Baseline tensor of the same shape on the specified device.
    """
    if baseline_type == 'black':
        return torch.zeros_like(images_tensor, device=device)
    elif baseline_type == 'blur':
        # Apply strong Gaussian blur to each image as a baseline
        blurred = images_tensor.detach().cpu().numpy().copy()
        for cam_idx in range(blurred.shape[0]):
            for ch in range(blurred.shape[1]):
                blurred[cam_idx, ch] = gaussian_filter(
                    blurred[cam_idx, ch], sigma=10.0
                )
        return torch.tensor(blurred, dtype=images_tensor.dtype, device=device)
    elif baseline_type == 'noise':
        # Random noise with same mean/std as the input
        noise = torch.randn_like(images_tensor, device=device)
        # Scale noise to match input distribution roughly
        noise = noise * 0.1
        return noise
    else:
        logger.warning(
            "Unknown baseline '%s', falling back to black (zeros).", baseline_type
        )
        return torch.zeros_like(images_tensor, device=device)


def _build_captum_forward(model, sample: dict, cell_i: int, cell_j: int,
                          class_idx: int, device: str):
    """Build a Captum-compatible forward function.

    Captum expects: forward(input_tensor) -> scalar tensor.
    The input_tensor here is the [6, 3, H, W] images tensor (treated as a
    single input with batch dim = 1 by adding a leading dim).

    Returns:
        A callable that takes images_tensor [1, 6, 3, H, W] and returns a scalar.
    """
    from pipeline.wrapper import forward_fn

    def captum_forward(images_batch: torch.Tensor) -> torch.Tensor:
        # images_batch: [1, 6, 3, H, W] from Captum (adds batch dim)
        # Squeeze out the batch dimension to get [6, 3, H, W]
        images = images_batch.squeeze(0)

        # Build a modified sample dict with the perturbed images
        modified_sample = dict(sample)
        modified_sample['image_tensors'] = images

        scalar = forward_fn(model, modified_sample, cell_i, cell_j,
                            class_idx=class_idx, device=device)
        # Captum expects output[0] to be indexable — wrap scalar as [1] tensor
        return scalar.unsqueeze(0)

    return captum_forward


def attr_ig(model, sample: dict, cell_i: int, cell_j: int,
            class_idx: int = 0, n_steps: int = 50,
            baseline: str = 'black', device: str = 'cpu') -> np.ndarray:
    """Compute Integrated Gradients attribution using Captum.

    Args:
        model: The loaded BEV model.
        sample: Sample dict containing 'image_tensors' [6, 3, H, W] and other fields.
        cell_i: BEV grid row index.
        cell_j: BEV grid column index.
        class_idx: Class channel index in the BEV output.
        n_steps: Number of interpolation steps for IG.
        baseline: Baseline type: 'black', 'blur', or 'noise'.
        device: Torch device string.

    Returns:
        [6, H, W] numpy array of attribution heatmaps, normalized to [0, 1].
    """
    from .utils import postprocess_heatmap

    try:
        from captum.attr import IntegratedGradients
    except ImportError as e:
        raise ImportError(
            "Captum is required for Integrated Gradients. "
            "Install it with: pip install captum"
        ) from e

    try:
        images_tensor = sample['image_tensors']  # [6, 3, H, W]
        if not isinstance(images_tensor, torch.Tensor):
            images_tensor = torch.tensor(images_tensor)

        images_tensor = images_tensor.to(device).float()
        images_tensor.requires_grad_(True)

        # Captum operates on inputs with a batch dimension
        # We add dim 0: [1, 6, 3, H, W]
        input_tensor = images_tensor.unsqueeze(0)

        # Create the baseline
        baseline_tensor = _make_baseline(images_tensor, baseline, device)
        baseline_tensor = baseline_tensor.unsqueeze(0)  # [1, 6, 3, H, W]

        # Build the forward function for Captum
        captum_fwd = _build_captum_forward(
            model, sample, cell_i, cell_j, class_idx, device
        )

        # Run Integrated Gradients
        ig = IntegratedGradients(captum_fwd)
        attributions = ig.attribute(
            input_tensor,
            baselines=baseline_tensor,
            n_steps=n_steps,
            method='gausslegendre',
        )

        # attributions shape: [1, 6, 3, H, W]
        # Sum over the channel dimension to get per-pixel importance
        attr_np = attributions.squeeze(0).detach().cpu().numpy()  # [6, 3, H, W]
        # Sum across color channels -> [6, H, W]
        attr_np = np.sum(np.abs(attr_np), axis=1)

        # Post-process: smooth and normalize
        result = postprocess_heatmap(attr_np, sigma=3.0)

        logger.info(
            "Integrated Gradients attribution computed: shape=%s, "
            "n_steps=%d, baseline='%s'",
            result.shape, n_steps, baseline,
        )
        return result

    except Exception as e:
        logger.error("Integrated Gradients failed: %s", str(e), exc_info=True)
        # Return zero heatmaps matching expected shape
        h, w = sample['image_tensors'].shape[-2:]
        logger.warning(
            "Returning zero heatmaps of shape [6, %d, %d] due to IG failure.", h, w
        )
        return np.zeros((6, h, w), dtype=np.float64)
