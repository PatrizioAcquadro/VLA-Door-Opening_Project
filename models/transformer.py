"""Transformer model implementation."""

import torch
import torch.nn as nn
from omegaconf import DictConfig


class TransformerModel(nn.Module):
    """Transformer model for sequence modeling.

    Args:
        cfg: Model configuration from Hydra
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        arch = cfg.model.architecture

        self.hidden_size = arch.hidden_size
        self.num_layers = arch.num_layers
        self.num_heads = arch.num_attention_heads
        self.max_seq_length = arch.max_seq_length

        # Input projection (assuming continuous state input)
        # Adjust input_dim based on your actual data
        self.input_proj = nn.Linear(256, self.hidden_size)  # 256 = state_dim placeholder

        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_seq_length, self.hidden_size))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=arch.intermediate_size,
            dropout=arch.hidden_dropout,
            activation=arch.activation,
            batch_first=True,
            norm_first=arch.use_pre_norm,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
        )

        # Output projection
        self.output_proj = nn.Linear(self.hidden_size, 256)  # Back to state_dim

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights."""
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # Initialize projections
        for module in [self.input_proj, self.output_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: Input tensor [batch, seq_len, state_dim]
            attention_mask: Optional mask [batch, seq_len]

        Returns:
            Dictionary with 'logits' and optionally other outputs
        """
        batch_size, seq_len, _ = input_ids.shape

        # Project input to hidden dimension
        hidden = self.input_proj(input_ids)

        # Add positional encoding
        hidden = hidden + self.pos_embedding[:, :seq_len, :]

        # Create causal mask for autoregressive modeling
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=input_ids.device
        )

        # Create padding mask from attention_mask
        padding_mask = None
        if attention_mask is not None:
            padding_mask = attention_mask == 0  # True = masked

        # Transformer encoding
        hidden = self.encoder(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )

        # Project to output
        logits = self.output_proj(hidden)

        return {"logits": logits, "hidden_states": hidden}

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute MSE loss for state prediction.

        Args:
            logits: Model predictions [batch, seq_len, state_dim]
            labels: Ground truth [batch, seq_len, state_dim]
            attention_mask: Optional mask [batch, seq_len]

        Returns:
            Scalar loss tensor
        """
        # Shift for next-state prediction
        logits = logits[:, :-1, :]
        labels = labels[:, 1:, :]

        if attention_mask is not None:
            mask = attention_mask[:, 1:].unsqueeze(-1)
            loss = ((logits - labels) ** 2 * mask).sum() / mask.sum() / logits.shape[-1]
        else:
            loss = nn.functional.mse_loss(logits, labels)

        return loss
