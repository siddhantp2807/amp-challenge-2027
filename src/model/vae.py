import torch
import torch.nn as nn

from src.data.tokenizer import VOCAB
from src.model.decoder import NonARTransformerDecoder
from src.model.encoder import TransformerEncoderVAE


class SequenceVAE(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        encoder_layers: int,
        decoder_layers: int,
        max_len: int,
        d_z: int,
        decoder_d_model: int | None = None,
        decoder_nhead: int | None = None,
        decoder_dim_feedforward: int | None = None,
        length_head: str = "regression",
        length_hidden: int = 256,
    ):
        super().__init__()
        vocab_size = len(VOCAB)
        self.d_z = d_z
        self.max_len = max_len

        # decoder capacity is independent of the encoder's -- there's no
        # cross-attention tying their dimensions together, only a shared
        # config by convention. Defaults preserve old configs/checkpoints
        # that only ever set the encoder-named keys.
        decoder_d_model = decoder_d_model or d_model
        decoder_nhead = decoder_nhead or nhead
        decoder_dim_feedforward = decoder_dim_feedforward or dim_feedforward

        self.encoder = TransformerEncoderVAE(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=encoder_layers,
            dropout=dropout,
            max_len=max_len,
            d_z=d_z,
        )
        self.decoder = NonARTransformerDecoder(
            vocab_size=vocab_size,
            d_model=decoder_d_model,
            nhead=decoder_nhead,
            dim_feedforward=decoder_dim_feedforward,
            num_layers=decoder_layers,
            dropout=dropout,
            max_len=max_len,
            d_z=d_z,
        )

        # latent normalization, populated once by freeze/export_frozen.py; a
        # pure affine pre/post transform kept outside the trained weights.
        self.register_buffer("latent_mean", torch.zeros(d_z))
        self.register_buffer("latent_std", torch.ones(d_z))

        # length-prediction head: z -> predicted AA length. Lets eval/generation
        # decode truncate to a predicted length instead of relying on the
        # non-AR decoder to reliably self-terminate at positions bucketed
        # training rarely covers (see src/train/losses.py's length loss).
        #   "regression": Linear(d_z, 1) on length/50, MSE (v1-v3 checkpoints)
        #   "classifier": MLP over the raw_max_length possible lengths, cross-
        #                 entropy; exact length is what bounds exact-match rate,
        #                 so this is trained and read out as a classification.
        self.raw_max_length = max_len - 2  # AA length excluding BOS/EOS
        self.length_head_type = length_head
        if length_head == "regression":
            self.length_head = nn.Linear(d_z, 1)
        elif length_head == "classifier":
            self.length_head = nn.Sequential(
                nn.Linear(d_z, length_hidden), nn.GELU(),
                nn.Linear(length_hidden, length_hidden), nn.GELU(),
                nn.Linear(length_hidden, self.raw_max_length),
            )
        else:
            raise ValueError(f"unknown length_head {length_head!r}")

    def encode(
        self, tokens: torch.Tensor, pad_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(tokens, pad_mask)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor, seq_len: int | None = None) -> torch.Tensor:
        return self.decoder(z, seq_len=seq_len)

    def forward(
        self, tokens: torch.Tensor, pad_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(tokens, pad_mask)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, seq_len=tokens.shape[1])
        return logits, mu, logvar

    def length_logits(self, z: torch.Tensor) -> torch.Tensor:
        """(B, raw_max_length) logits; index i is length i + 1. Classifier head only."""
        return self.length_head(z)

    def predict_length_frac(self, z: torch.Tensor) -> torch.Tensor:
        return self.length_head(z).squeeze(-1)  # (B,), regression head only

    def predict_length(self, z: torch.Tensor) -> torch.Tensor:
        if self.length_head_type == "classifier":
            return self.length_logits(z).argmax(dim=-1).float() + 1
        frac = self.predict_length_frac(z)
        return (frac * self.raw_max_length).round().clamp(1, self.raw_max_length)

    def normalize(self, mu: torch.Tensor) -> torch.Tensor:
        return (mu - self.latent_mean) / self.latent_std

    def denormalize(self, z_norm: torch.Tensor) -> torch.Tensor:
        return z_norm * self.latent_std + self.latent_mean
