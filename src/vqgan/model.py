"""Full VQGAN model combining encoder, quantizer, and decoder."""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple

from .encoder import Encoder
from .decoder import Decoder
from .codebook import VectorQuantizer


class VQGAN(nn.Module):
    """Vector Quantized GAN for pixel art compression.

    Encodes images to discrete tokens and decodes back to images.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 128,
        codebook_size: int = 512,
        codebook_dim: int = 256,
        num_res_blocks: int = 2
    ):
        super().__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            codebook_dim=codebook_dim,
            num_res_blocks=num_res_blocks
        )

        self.quantizer = VectorQuantizer(
            codebook_size=codebook_size,
            codebook_dim=codebook_dim
        )

        self.decoder = Decoder(
            out_channels=in_channels,
            hidden_channels=hidden_channels,
            codebook_dim=codebook_dim,
            num_res_blocks=num_res_blocks
        )

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """Forward pass: encode, quantize, decode.

        Args:
            x: Input image (B, C, H, W) in [-1, 1]

        Returns:
            x_recon: Reconstructed image (B, C, H, W)
            indices: Codebook indices (B, H', W')
            vq_loss: Vector quantization loss
        """
        z = self.encoder(x)
        z_q, indices, vq_loss = self.quantizer(z)
        x_recon = self.decoder(z_q)

        return x_recon, indices, vq_loss

    def encode(self, x: mx.array) -> mx.array:
        """Encode image to discrete indices.

        Args:
            x: Input image (B, C, H, W)

        Returns:
            indices: Codebook indices (B, H', W')
        """
        z = self.encoder(x)
        _, indices, _ = self.quantizer(z)
        return indices

    def decode(self, indices: mx.array) -> mx.array:
        """Decode indices to image.

        Args:
            indices: Codebook indices (B, H', W')

        Returns:
            x: Reconstructed image (B, C, H, W)
        """
        z_q = self.quantizer.decode(indices)
        x = self.decoder(z_q)
        return x
