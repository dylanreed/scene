"""Image token transformer for text-to-image generation."""

import mlx.core as mx
import mlx.nn as nn
import math
from typing import Optional

from .text_encoder import TextEncoder


class ImageTransformer(nn.Module):
    """Transformer that generates image tokens conditioned on text.

    Uses cross-attention to condition on text embeddings and
    generates image tokens autoregressively.
    """

    def __init__(
        self,
        vocab_size: int = 512,       # VQGAN codebook size
        text_vocab_size: int = 1000,
        max_seq_len: int = 240,      # 20x12 image tokens
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        text_embed_dim: int = 256,
        max_text_len: int = 128
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        # Text encoder
        self.text_encoder = TextEncoder(
            vocab_size=text_vocab_size,
            embed_dim=text_embed_dim,
            max_length=max_text_len
        )

        # Project text embeddings to match image embedding dimension if needed
        if text_embed_dim != embed_dim:
            self.text_proj = nn.Linear(text_embed_dim, embed_dim)
        else:
            self.text_proj = None

        # Image token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Transformer decoder layers with cross-attention
        self.layers = []
        for _ in range(num_layers):
            self.layers.append({
                'self_attn': nn.MultiHeadAttention(embed_dim, num_heads),
                'cross_attn': nn.MultiHeadAttention(embed_dim, num_heads),
                'mlp': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * mlp_ratio),
                    nn.GELU(),
                    nn.Linear(embed_dim * mlp_ratio, embed_dim)
                ),
                'norm1': nn.LayerNorm(embed_dim),
                'norm2': nn.LayerNorm(embed_dim),
                'norm3': nn.LayerNorm(embed_dim),
            })

        self.final_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def _causal_mask(self, seq_len: int) -> mx.array:
        """Create causal attention mask."""
        mask = mx.triu(mx.full((seq_len, seq_len), float('-inf')), k=1)
        return mask

    def forward_with_cache(
        self,
        text_embeddings: mx.array,
        image_tokens: mx.array,
        cache: Optional[list] = None
    ) -> tuple:
        """Forward pass with KV cache for efficient generation."""
        B, L = image_tokens.shape

        # Embed image tokens
        positions = mx.arange(L)
        x = self.token_embedding(image_tokens) + self.pos_embedding(positions)

        # Causal mask for self-attention
        causal_mask = self._causal_mask(L)

        new_cache = []
        for i, layer in enumerate(self.layers):
            # Self-attention with causal mask
            residual = x
            x = layer['norm1'](x)
            x = layer['self_attn'](x, x, x, mask=causal_mask) + residual

            # Cross-attention to text
            residual = x
            x = layer['norm2'](x)
            x = layer['cross_attn'](x, text_embeddings, text_embeddings) + residual

            # MLP
            residual = x
            x = layer['norm3'](x)
            x = layer['mlp'](x) + residual

            new_cache.append(None)  # Placeholder for future cache implementation

        x = self.final_norm(x)
        logits = self.output_proj(x)

        return logits, new_cache

    def __call__(
        self,
        text_tokens: mx.array,
        image_tokens: mx.array
    ) -> mx.array:
        """Forward pass for training.

        Args:
            text_tokens: Text token IDs (B, text_len)
            image_tokens: Image token IDs (B, seq_len)

        Returns:
            logits: Token logits (B, seq_len, vocab_size)
        """
        # Encode text
        text_embeddings = self.text_encoder(text_tokens)

        # Project text embeddings if needed
        if self.text_proj is not None:
            text_embeddings = self.text_proj(text_embeddings)

        # Generate logits
        logits, _ = self.forward_with_cache(text_embeddings, image_tokens)

        return logits

    def generate(
        self,
        text_tokens: mx.array,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> mx.array:
        """Generate image tokens autoregressively.

        Args:
            text_tokens: Text token IDs (B, text_len)
            temperature: Sampling temperature
            top_k: If set, only sample from top k tokens

        Returns:
            generated: Generated image tokens (B, max_seq_len)
        """
        B = text_tokens.shape[0]

        # Encode text once
        text_embeddings = self.text_encoder(text_tokens)

        # Project text embeddings if needed
        if self.text_proj is not None:
            text_embeddings = self.text_proj(text_embeddings)

        # Start with a random first token (or learned start token)
        generated = mx.zeros((B, 1), dtype=mx.int32)

        for i in range(self.max_seq_len - 1):
            logits, _ = self.forward_with_cache(text_embeddings, generated)
            next_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                # Top-k sampling
                top_k_vals, top_k_idx = mx.topk(next_logits, k=top_k)
                probs = mx.softmax(top_k_vals, axis=-1)
                sampled_idx = mx.random.categorical(probs)
                next_token = mx.take_along_axis(top_k_idx, sampled_idx[:, None], axis=-1)
            else:
                # Full sampling
                probs = mx.softmax(next_logits, axis=-1)
                next_token = mx.random.categorical(probs)[:, None]

            generated = mx.concatenate([generated, next_token.astype(mx.int32)], axis=1)

        return generated
