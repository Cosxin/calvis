"""Custom pytorch-grad-cam targets for VLM token attribution."""


class VLMTokenTarget:
    """Target that extracts the logit for a specific generated token.

    pytorch-grad-cam calls target(model_output) and expects a scalar.
    We index into the logits at the position just before the target token
    (teacher-forced), extracting the logit for the token that was actually
    generated.
    """

    def __init__(self, token_position: int, token_id: int):
        """
        Args:
            token_position: position in the full sequence whose logit we want
                (the position that predicts token_id as next token).
            token_id: the vocabulary ID of the generated token.
        """
        self.token_position = token_position
        self.token_id = token_id

    def __call__(self, model_output):
        # pytorch-grad-cam iterates over the batch and passes each element
        # separately, so model_output is [seq_len, vocab_size] (2D, no batch dim)
        if model_output.ndim == 3:
            return model_output[0, self.token_position, self.token_id]
        return model_output[self.token_position, self.token_id]
