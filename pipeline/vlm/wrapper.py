"""Model wrapper that makes a VLM look like image→scalar for pytorch-grad-cam."""

import torch
import torch.nn as nn


class SmolVLMForCAM(nn.Module):
    """Wraps SmolVLM so pytorch-grad-cam can attribute image patches to a token logit.

    pytorch-grad-cam expects: model(input_tensor) → output
    where input_tensor is the thing we want gradients w.r.t. (pixel_values),
    and output is what the target callable indexes into (logits).

    This wrapper freezes everything except pixel_values as the input,
    teacher-forces the generated sequence, and returns the full logits.
    """

    def __init__(self, model, processor, input_ids, attention_mask,
                 pixel_values_shape):
        super().__init__()
        self.model = model
        self.processor = processor
        self._input_ids = input_ids
        self._attention_mask = attention_mask
        self._pixel_values_shape = pixel_values_shape

    def forward(self, pixel_values):
        """Forward pass: pixel_values → logits.

        Args:
            pixel_values: [B, C, H, W] from pytorch-grad-cam (4D).

        Returns:
            logits: [B, seq_len, vocab_size] tensor.
        """
        # pytorch-grad-cam passes [B, C, H, W] but SmolVLM expects
        # [B, num_patches, C, H, W] — add the patches dimension
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(1)  # [B, 1, C, H, W]
        outputs = self.model(
            input_ids=self._input_ids,
            attention_mask=self._attention_mask,
            pixel_values=pixel_values,
        )
        return outputs.logits
