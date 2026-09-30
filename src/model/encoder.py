import torch
import torch.nn as nn

from src.data.tokenizer import PAD_ID
from src.model.positional import LearnedPositionalEmbedding


class TransformerEncoderVAE(nn.Module):
    """sequence -> (mu, logvar) for a continuous latent z.

    Attention pooling (a single learned query attending over non-pad
    positions) over mean/CLS pooling: attention weights are inspectable
    (feeds the density/holes diagnostic) and the model can down-weight
    uninformative positions instead of forcing uniform contribution, which
    gives smoother pooled representations for variable-length inputs.
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
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
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

        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )

        self.to_mu = nn.Linear(d_model, d_z)
        self.to_logvar = nn.Linear(d_model, d_z)

        self.last_pool_weights: torch.Tensor | None = None

    def forward(
        self, tokens: torch.Tensor, pad_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """tokens: (B, L) long, pad_mask: (B, L) bool, True where padded."""
        B, L = tokens.shape
        x = self.token_embedding(tokens) + self.positional(L).unsqueeze(0)
        h = self.backbone(x, src_key_padding_mask=pad_mask)

        query = self.pool_query.expand(B, -1, -1)
        pooled, attn_weights = self.pool_attn(
            query, h, h, key_padding_mask=pad_mask, need_weights=True
        )
        self.last_pool_weights = attn_weights.detach()  # (B, 1, L), for diagnostics
        pooled = pooled.squeeze(1)  # (B, d_model)

        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled)
        return mu, logvar
