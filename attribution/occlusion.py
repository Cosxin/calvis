"""
Occlusion-based attribution for BEV models using Captum.
"""

import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


def _build_captum_forward(model, sample: dict, cell_i: int, cell_j: int,
                          class_idx: int, device: str):
    """Build a Captum-compatible forward function for Occlusion.

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


def attr_occlusion(model, sample: dict, cell_i: int, cell_j: int,
                   class_idx: int = 0, patch_size: int = 32,
                   stride: int = 16, device: str = 'cpu') -> np.ndarray:
    """Compute occlusion-based attribution using Captum.

    Slides a grey patch across each camera image and measures the change
    in the model's output at the target BEV cell. Regions where occlusion
    causes a large drop in the output are considered important.

    Args:
        model: The loaded BEV model.
        sample: Sample dict containing 'image_tensors' [6, 3, H, W].
        cell_i: BEV grid row index.
        cell_j: BEV grid column index.
        class_idx: Class channel index in the BEV output.
        patch_size: Size of the occluding square patch in pixels.
        stride: Stride for sliding the patch across the image.
        device: Torch device string.

    Returns:
        [6, H, W] numpy array of attribution heatmaps, normalized to [0, 1].
    """
    from .utils import postprocess_heatmap

    try:
        from captum.attr import Occlusion
    except ImportError as e:
        raise ImportError(
            "Captum is required for Occlusion attribution. "
            "Install it with: pip install captum"
        ) from e

    try:
        images_tensor = sample['image_tensors']  # [6, 3, H, W]
        if not isinstance(images_tensor, torch.Tensor):
            images_tensor = torch.tensor(images_tensor)
        images_tensor = images_tensor.to(device).float()

        n_cams, n_channels, img_h, img_w = images_tensor.shape

        # Captum Occlusion on all cameras at once can be very slow.
        # Instead, we run occlusion per camera for efficiency and clarity.
        all_heatmaps = np.zeros((n_cams, img_h, img_w), dtype=np.float64)

        # Grey baseline value (mid-range for normalized images)
        # Use 0.5 as a neutral grey for normalized inputs, or 0.0 if
        # the images are zero-centered.
        img_mean = images_tensor.mean().item()
        baseline_val = 0.5 if img_mean > -0.5 else 0.0

        captum_fwd = _build_captum_forward(
            model, sample, cell_i, cell_j, class_idx, device
        )

        # Run occlusion with the full [1, 6, 3, H, W] input
        input_tensor = images_tensor.unsqueeze(0)  # [1, 6, 3, H, W]

        occ = Occlusion(captum_fwd)

        # Sliding window configuration:
        # - The window slides over the [6, 3, H, W] dimensions (after squeezing batch)
        # - We want to occlude spatial patches: window = (1, 3, patch_size, patch_size)
        #   so that each patch covers all 3 channels of one camera at a time
        # - Strides: (1, 3, stride, stride) - move one camera at a time, all channels
        sliding_window = (1, n_channels, patch_size, patch_size)
        strides = (1, n_channels, stride, stride)

        # Baseline: grey patch
        baselines = torch.full_like(input_tensor, baseline_val)

        attributions = occ.attribute(
            input_tensor,
            sliding_window_shapes=sliding_window,
            strides=strides,
            baselines=baselines,
            show_progress=False,
        )

        # attributions: [1, 6, 3, H, W]
        attr_np = attributions.squeeze(0).detach().cpu().numpy()  # [6, 3, H, W]

        # Sum over channels
        all_heatmaps = np.sum(np.abs(attr_np), axis=1)  # [6, H, W]

        result = postprocess_heatmap(all_heatmaps, sigma=3.0)

        logger.info(
            "Occlusion attribution computed: shape=%s, "
            "patch_size=%d, stride=%d",
            result.shape, patch_size, stride,
        )
        return result

    except Exception as e:
        logger.error("Occlusion attribution failed: %s", str(e), exc_info=True)

        # Try per-camera fallback
        return _occlusion_per_camera_fallback(
            model, sample, cell_i, cell_j, class_idx,
            patch_size, stride, device
        )


def _occlusion_per_camera_fallback(model, sample: dict, cell_i: int, cell_j: int,
                                   class_idx: int, patch_size: int,
                                   stride: int, device: str) -> np.ndarray:
    """Per-camera occlusion fallback: manually slide patches one camera at a time.

    This is slower but more robust when Captum's Occlusion has trouble with
    the 5D input shape.
    """
    from .utils import postprocess_heatmap
    from pipeline.wrapper import forward_fn

    try:
        images_tensor = sample['image_tensors']
        if not isinstance(images_tensor, torch.Tensor):
            images_tensor = torch.tensor(images_tensor)
        images_tensor = images_tensor.to(device).float()

        n_cams, n_channels, img_h, img_w = images_tensor.shape
        all_heatmaps = np.zeros((n_cams, img_h, img_w), dtype=np.float64)

        # Get the unoccluded baseline score
        with torch.no_grad():
            baseline_score = forward_fn(
                model, sample, cell_i, cell_j,
                class_idx=class_idx, device=device
            ).item()

        # Grey value for occlusion
        baseline_val = 0.5

        for cam_idx in range(n_cams):
            logger.debug("Occlusion: processing camera %d/%d", cam_idx + 1, n_cams)

            for y in range(0, img_h, stride):
                for x in range(0, img_w, stride):
                    # Create occluded input
                    occluded = images_tensor.clone()
                    y_end = min(y + patch_size, img_h)
                    x_end = min(x + patch_size, img_w)
                    occluded[cam_idx, :, y:y_end, x:x_end] = baseline_val

                    modified_sample = dict(sample)
                    modified_sample['image_tensors'] = occluded

                    with torch.no_grad():
                        score = forward_fn(
                            model, modified_sample, cell_i, cell_j,
                            class_idx=class_idx, device=device
                        ).item()

                    # Attribution = how much the score drops when this patch is occluded
                    importance = baseline_score - score
                    all_heatmaps[cam_idx, y:y_end, x:x_end] += abs(importance)

            # Normalize by overlap count (each pixel may be covered by multiple patches)
            count_map = np.zeros((img_h, img_w), dtype=np.float64)
            for y in range(0, img_h, stride):
                for x in range(0, img_w, stride):
                    y_end = min(y + patch_size, img_h)
                    x_end = min(x + patch_size, img_w)
                    count_map[y:y_end, x:x_end] += 1.0
            count_map = np.maximum(count_map, 1.0)
            all_heatmaps[cam_idx] /= count_map

        result = postprocess_heatmap(all_heatmaps, sigma=3.0)

        logger.info(
            "Occlusion attribution (per-camera fallback) computed: shape=%s",
            result.shape,
        )
        return result

    except Exception as e:
        logger.error("Occlusion per-camera fallback failed: %s", str(e), exc_info=True)
        h, w = sample['image_tensors'].shape[-2:]
        logger.warning(
            "Returning zero heatmaps of shape [6, %d, %d] due to occlusion failure.",
            h, w,
        )
        return np.zeros((6, h, w), dtype=np.float64)
