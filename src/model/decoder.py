import torch
import torch.nn as nn

from src.model.positional import LearnedPositionalEmbedding


class NonARTransformerDecoder(nn.Module):
    """z -> per-position token logits, non-autoregressive.

    No causal mask: every output position attends to every other output
    position (self-attention only). This avoids exposure-bias tuning and
    keeps the posterior-collapse / perturbation-sensitivity checks clean —
    output differences are attributable to z alone, not autoregressive
    sampling noise. z is broadcast to a fixed max length and the model
    decodes to EOS; length is never a discrete side-channel of z itself.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float,
        max_len: int,
        d_z: int,
    ):
        super().__init__()
        self.max_len = max_len
        self.z_to_model = nn.Linear(d_z, d_model)
        self.positional = LearnedPositionalEmbedding(max_len, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.to_logits = nn.Linear(d_model, vocab_size)

    def forward(self, z: torch.Tensor, seq_len: int | None = None) -> torch.Tensor:
        """z: (B, d_z). Returns logits (B, seq_len, vocab_size)."""
        L = seq_len or self.max_len
        z_proj = self.z_to_model(z).unsqueeze(1)  # (B, 1, d_model)
        x = z_proj.expand(-1, L, -1) + self.positional(L).unsqueeze(0)
        h = self.backbone(x)
        return self.to_logits(h)
