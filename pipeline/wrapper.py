"""
Captum-compatible model wrapper for the BEV Attribution Debug Tool.

Provides two main functions:

  - ``infer(model, sample)`` — run full inference, return the BEV logit grid.
  - ``forward_fn(model, sample, cell_i, cell_j, class_idx)`` — return a single
    scalar (the logit at a specific BEV cell) with gradients enabled, suitable
    for use with Captum attribution methods.
"""

import logging
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _prepare_input(
    sample: dict, device: str = "cpu"
) -> torch.Tensor:
    """Extract the image tensor from a sample dict and prepare it for the model.

    Returns a [1, 6, 3, H, W] float32 tensor on ``device`` with
    ``requires_grad=False`` (the attribution methods will set grad as needed).
    """
    tensor = sample["image_tensors"]  # [6, 3, H, W]
    if tensor.dim() == 4:
        tensor = tensor.unsqueeze(0)  # [1, 6, 3, H, W]
    return tensor.to(device=device, dtype=torch.float32)


def _run_model(
    model: nn.Module,
    images: torch.Tensor,
    sample: Optional[dict] = None,
) -> torch.Tensor:
    """Run the model forward pass.

    Handles SimpleBEVModel, LSSWrapper, and MMDet3D models.
    For LSSWrapper, calibration is injected from the sample dict.

    Returns:
        [B, C, H_bev, W_bev] logit grid.
    """
    # Inject calibration for LSS-style models that need it.
    if hasattr(model, "set_calibration") and sample is not None:
        model.set_calibration(sample)
    return model(images)


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #

def infer(
    model: nn.Module,
    sample: dict,
    device: str = "cpu",
) -> np.ndarray:
    """Run inference and return the BEV logit grid as a numpy array.

    Parameters
    ----------
    model : nn.Module
        BEV model (SimpleBEVModel or MMDet3D model) in eval mode.
    sample : dict
        Output of ``pipeline.data.load_sample()``.
    device : str
        Device to run on ('cpu', 'cuda', etc.).

    Returns
    -------
    np.ndarray of shape [C, H_bev, W_bev] — per-cell class logits.
    """
    model.eval()
    images = _prepare_input(sample, device)

    with torch.no_grad():
        logits = _run_model(model, images, sample=sample)  # [1, C, H, W]

    return logits.squeeze(0).cpu().numpy()  # [C, H, W]


def forward_fn(
    model: nn.Module,
    sample: dict,
    cell_i: int,
    cell_j: int,
    class_idx: int = 0,
    device: str = "cpu",
) -> torch.Tensor:
    """Captum-compatible forward function: images → scalar.

    Runs the model on the sample's images and extracts the logit at
    ``[class_idx, cell_i, cell_j]`` from the BEV grid.  The returned tensor
    is a differentiable scalar with ``requires_grad`` support through the
    full computation graph.

    Parameters
    ----------
    model : nn.Module
        BEV model.  Should be in eval mode but with parameters that allow
        gradient flow (the default after ``model.eval()``).
    sample : dict
        Output of ``pipeline.data.load_sample()``.
    cell_i : int
        Row index into the BEV grid (0 .. H_bev-1).
    cell_j : int
        Column index into the BEV grid (0 .. W_bev-1).
    class_idx : int
        Class channel to extract (0 .. num_classes-1).
    device : str
        Device to run on.

    Returns
    -------
    torch.Tensor — scalar (shape ``[]``) with grad graph attached.
    """
    images = _prepare_input(sample, device)
    images = images.requires_grad_(True)

    # Inject calibration for LSS-style models.
    if hasattr(model, "set_calibration"):
        model.set_calibration(sample)

    logits = _run_model(model, images)  # [1, C, H_bev, W_bev]

    # Validate indices.
    _, C, H, W = logits.shape
    if class_idx < 0 or class_idx >= C:
        raise IndexError(
            f"class_idx={class_idx} out of range [0, {C - 1}]"
        )
    if cell_i < 0 or cell_i >= H:
        raise IndexError(f"cell_i={cell_i} out of range [0, {H - 1}]")
    if cell_j < 0 or cell_j >= W:
        raise IndexError(f"cell_j={cell_j} out of range [0, {W - 1}]")

    scalar = logits[0, class_idx, cell_i, cell_j]
    return scalar


def make_captum_forward(
    model: nn.Module,
    cell_i: int,
    cell_j: int,
    class_idx: int = 0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create a closure suitable for Captum's ``IntegratedGradients(forward_func)``.

    Captum expects ``forward_func(inputs) -> scalar`` where ``inputs`` is
    exactly the tensor whose attribution is desired.  This factory returns
    such a function with the cell/class indices baked in.

    Usage::

        from captum.attr import IntegratedGradients

        fn = make_captum_forward(model, cell_i=100, cell_j=100, class_idx=0)
        ig = IntegratedGradients(fn)
        images = sample["image_tensors"].unsqueeze(0).requires_grad_(True)
        attr = ig.attribute(images, baselines=torch.zeros_like(images))
        # attr shape: [1, 6, 3, H, W]

    Parameters
    ----------
    model : nn.Module
        BEV model in eval mode.
    cell_i, cell_j : int
        BEV grid cell coordinates.
    class_idx : int
        Class channel index.

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        ``forward_func(images: [B, 6, 3, H, W]) -> scalar``
    """
    model.eval()

    def _forward(images: torch.Tensor) -> torch.Tensor:
        """images: [B, 6, 3, H, W] → scalar."""
        logits = model(images)  # [B, C, H_bev, W_bev]
        return logits[0, class_idx, cell_i, cell_j]

    return _forward


def make_captum_forward_batched(
    model: nn.Module,
    cell_i: int,
    cell_j: int,
    class_idx: int = 0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Like ``make_captum_forward`` but returns [B] instead of scalar.

    Some Captum methods (e.g., ``Occlusion``) need the forward function to
    return one value per batch element.
    """
    model.eval()

    def _forward(images: torch.Tensor) -> torch.Tensor:
        """images: [B, 6, 3, H, W] → [B]."""
        logits = model(images)  # [B, C, H_bev, W_bev]
        return logits[:, class_idx, cell_i, cell_j]  # [B]

    return _forward
