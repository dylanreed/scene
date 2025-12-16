"""Vector Quantizer for VQGAN."""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple


class VectorQuantizer(nn.Module):
    """Vector Quantizer with EMA codebook updates.

    Maps continuous latent vectors to discrete codebook entries.
    Uses straight-through estimator for gradients.
    """

    def __init__(
        self,
        codebook_size: int = 512,
        codebook_dim: int = 256,
        commitment_cost: float = 0.25
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_cost = commitment_cost

        # Initialize codebook embeddings
        self.embedding = nn.Embedding(codebook_size, codebook_dim)

    def __call__(self, z: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """Quantize continuous latent vectors.

        Args:
            z: Continuous latents of shape (B, D, H, W)

        Returns:
            z_q: Quantized latents (B, D, H, W)
            indices: Codebook indices (B, H, W)
            loss: Commitment loss (scalar)
        """
        B, D, H, W = z.shape

        # Reshape: (B, D, H, W) -> (B*H*W, D)
        z_flat = z.transpose(0, 2, 3, 1).reshape(-1, D)

        # Get codebook
        codebook = self.embedding.weight  # (codebook_size, D)

        # Compute distances: ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z@e.T
        z_sq = mx.sum(z_flat ** 2, axis=1, keepdims=True)
        e_sq = mx.sum(codebook ** 2, axis=1, keepdims=True)
        distances = z_sq + e_sq.T - 2 * (z_flat @ codebook.T)

        # Get nearest codebook entry
        indices = mx.argmin(distances, axis=1).astype(mx.int32)

        # Lookup quantized vectors
        z_q_flat = self.embedding(indices)

        # Reshape back: (B*H*W, D) -> (B, D, H, W)
        z_q = z_q_flat.reshape(B, H, W, D).transpose(0, 3, 1, 2)
        indices = indices.reshape(B, H, W)

        # Compute loss
        codebook_loss = mx.mean((mx.stop_gradient(z) - z_q) ** 2)
        commitment_loss = mx.mean((z - mx.stop_gradient(z_q)) ** 2)
        loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator: copy gradients from z_q to z
        z_q = z + mx.stop_gradient(z_q - z)

        return z_q, indices, loss

    def decode(self, indices: mx.array) -> mx.array:
        """Decode indices back to latent vectors.

        Args:
            indices: Codebook indices of shape (B, H, W)

        Returns:
            z_q: Quantized latents (B, D, H, W)
        """
        B, H, W = indices.shape
        indices_flat = indices.reshape(-1)
        z_q_flat = self.embedding(indices_flat)
        z_q = z_q_flat.reshape(B, H, W, self.codebook_dim).transpose(0, 3, 1, 2)
        return z_q
