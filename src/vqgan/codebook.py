# ABOUTME: Vector Quantizer for VQGAN with EMA updates and dead code reset
# ABOUTME: Maps continuous latent vectors to discrete codebook entries

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Optional


class VectorQuantizer(nn.Module):
    """Vector Quantizer with optional EMA codebook updates.

    Maps continuous latent vectors to discrete codebook entries.
    Uses straight-through estimator for gradients.
    Supports EMA updates for more stable codebook learning.
    """

    def __init__(
        self,
        codebook_size: int = 512,
        codebook_dim: int = 256,
        commitment_cost: float = 0.25,
        use_ema: bool = False,
        ema_decay: float = 0.99,
        reset_threshold: int = 2
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_cost = commitment_cost
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.reset_threshold = reset_threshold

        # Initialize codebook embeddings
        self.embedding = nn.Embedding(codebook_size, codebook_dim)

        # EMA tracking (not trainable parameters)
        self._ema_cluster_size = mx.zeros((codebook_size,))
        self._ema_embedding_sum = mx.zeros((codebook_size, codebook_dim))
        self._code_usage = mx.zeros((codebook_size,))

    def __call__(self, z: mx.array, training: bool = True) -> Tuple[mx.array, mx.array, mx.array]:
        """Quantize continuous latent vectors.

        Args:
            z: Continuous latents of shape (B, D, H, W)
            training: Whether in training mode (enables EMA updates)

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

        # Build one_hot once if needed for tracking or EMA
        one_hot = None
        if training:
            one_hot = mx.zeros((indices.shape[0], self.codebook_size))
            one_hot = one_hot.at[mx.arange(indices.shape[0]), indices].add(1.0)
            # Track code usage for dead code detection
            self._code_usage = self._code_usage + mx.sum(one_hot, axis=0)

        # EMA codebook updates during training
        if training and self.use_ema:
            # Reuse one_hot from above
            cluster_size = mx.sum(one_hot, axis=0)
            self._ema_cluster_size = (
                self.ema_decay * self._ema_cluster_size +
                (1 - self.ema_decay) * cluster_size
            )

            # Update embedding sums
            embedding_sum = one_hot.T @ z_flat
            self._ema_embedding_sum = (
                self.ema_decay * self._ema_embedding_sum +
                (1 - self.ema_decay) * embedding_sum
            )

            # Normalize to get new embeddings (with Laplace smoothing)
            n = mx.sum(self._ema_cluster_size)
            cluster_size_smoothed = (
                (self._ema_cluster_size + 1e-5) /
                (n + self.codebook_size * 1e-5) * n
            )
            new_embedding = self._ema_embedding_sum / cluster_size_smoothed[:, None]

            # Update embedding weights and force evaluation to prevent graph buildup
            self.embedding.weight = new_embedding
            mx.eval(self._ema_cluster_size, self._ema_embedding_sum, self.embedding.weight)

        # Lookup quantized vectors
        z_q_flat = self.embedding(indices)

        # Reshape back: (B*H*W, D) -> (B, D, H, W)
        z_q = z_q_flat.reshape(B, H, W, D).transpose(0, 3, 1, 2)
        indices = indices.reshape(B, H, W)

        # Compute loss
        if self.use_ema:
            # With EMA, only need commitment loss (codebook learns via EMA)
            loss = self.commitment_cost * mx.mean((z - mx.stop_gradient(z_q)) ** 2)
        else:
            # Standard VQ loss
            codebook_loss = mx.mean((mx.stop_gradient(z) - z_q) ** 2)
            commitment_loss = mx.mean((z - mx.stop_gradient(z_q)) ** 2)
            loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator: copy gradients from z_q to z
        z_q = z + mx.stop_gradient(z_q - z)

        return z_q, indices, loss

    def reset_dead_codes(self, encoder_outputs: mx.array) -> int:
        """Reset codebook entries that are rarely used.

        Args:
            encoder_outputs: Recent encoder outputs (N, D) to sample from

        Returns:
            Number of codes reset
        """
        # Find dead codes (used less than threshold)
        dead_mask = self._code_usage < self.reset_threshold
        num_dead = int(mx.sum(dead_mask).item())

        if num_dead > 0 and encoder_outputs.shape[0] > 0:
            # Sample random encoder outputs to replace dead codes
            import numpy as np
            dead_indices = np.where(np.array(dead_mask))[0]
            num_to_reset = min(num_dead, encoder_outputs.shape[0])

            # Random sample from encoder outputs
            sample_indices = mx.random.randint(0, encoder_outputs.shape[0], (num_to_reset,))
            new_embeddings = encoder_outputs[sample_indices]

            # Batch update: copy weights to numpy, modify, copy back (O(1) allocations)
            weight_np = np.array(self.embedding.weight)
            new_emb_np = np.array(new_embeddings)
            for i in range(num_to_reset):
                weight_np[dead_indices[i]] = new_emb_np[i]
            self.embedding.weight = mx.array(weight_np)

            # Batch update EMA state similarly
            if self.use_ema:
                ema_size_np = np.array(self._ema_cluster_size)
                ema_sum_np = np.array(self._ema_embedding_sum)
                for i in range(num_to_reset):
                    idx = dead_indices[i]
                    ema_size_np[idx] = 0.0
                    ema_sum_np[idx] = new_emb_np[i]
                self._ema_cluster_size = mx.array(ema_size_np)
                self._ema_embedding_sum = mx.array(ema_sum_np)

            # Force evaluation to free memory
            mx.eval(self.embedding.weight)
            if self.use_ema:
                mx.eval(self._ema_cluster_size, self._ema_embedding_sum)

        # Reset usage counter for next epoch
        self._code_usage = mx.zeros((self.codebook_size,))

        return num_dead

    def get_codebook_usage(self) -> Tuple[int, float]:
        """Get codebook utilization statistics.

        Returns:
            num_used: Number of codes used at least once
            avg_usage: Average usage per code
        """
        num_used = int(mx.sum(self._code_usage > 0).item())
        total_usage = float(mx.sum(self._code_usage).item())
        avg_usage = total_usage / self.codebook_size if self.codebook_size > 0 else 0
        return num_used, avg_usage

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
