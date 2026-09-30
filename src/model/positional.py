import torch
import torch.nn as nn


class LearnedPositionalEmbedding(nn.Module):
    """Learned (not sinusoidal) positional embedding — max_len is small and
    fixed (52), where learned embeddings are a defensible default and add
    negligible params. Shared between encoder input and decoder output so
    position semantics stay consistent across both.
    """

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, seq_len: int) -> torch.Tensor:
        positions = torch.arange(seq_len, device=self.embedding.weight.device)
        return self.embedding(positions)  # (seq_len, d_model)
