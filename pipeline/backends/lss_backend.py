"""LSS (Lift-Splat-Shoot) backend -- wraps existing LSSWrapper."""

import logging
import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .base import BEVBackend

logger = logging.getLogger(__name__)

CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints")


class LSSBackend(BEVBackend):
    name = "lss"
    repr_type = "bev_seg"

    @property
    def class_names(self) -> List[str]:
        return ["vehicle"]

    def load(self, device: str = "cpu") -> nn.Module:
        from pipeline.lss_model import create_lss_model
        ckpt = os.path.join(CKPT_DIR, "lss_model.pt")
        if not os.path.isfile(ckpt):
            logger.warning("LSS checkpoint not found at %s", ckpt)
            return None
        model = create_lss_model(checkpoint_path=ckpt, device=device)
        return model

    def infer(self, model: nn.Module, sample: dict, device: str = "cpu") -> np.ndarray:
        from pipeline.wrapper import infer as _infer
        return _infer(model, sample, device=device)

    def get_checkpoint_url(self) -> Optional[str]:
        return None  # Already downloaded

    def get_checkpoint_filename(self) -> str:
        return "lss_model.pt"
