"""Text encoder for conditioning the transformer."""

import mlx.core as mx
import mlx.nn as nn
import math


class TextEncoder(nn.Module):
    """Transformer-based text encoder.

    Encodes text tokens into contextual embeddings for
    conditioning the image generation transformer.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        max_length: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.max_length = max_length

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional embeddings
        self.pos_embedding = nn.Embedding(max_length, embed_dim)

        # Transformer layers
        self.layers = [
            nn.TransformerEncoderLayer(
                dims=embed_dim,
                num_heads=num_heads,
                mlp_dims=embed_dim * 4,
                dropout=dropout
            )
            for _ in range(num_layers)
        ]

        self.norm = nn.LayerNorm(embed_dim)

    def __call__(self, tokens: mx.array) -> mx.array:
        """Encode tokens to embeddings.

        Args:
            tokens: Token IDs (B, L)

        Returns:
            embeddings: Contextual embeddings (B, L, D)
        """
        B, L = tokens.shape

        # Token + positional embeddings
        positions = mx.arange(L)
        x = self.token_embedding(tokens) + self.pos_embedding(positions)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask=None)

        x = self.norm(x)

        return x
