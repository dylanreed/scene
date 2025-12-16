"""VQGAN Encoder - compresses images to continuous latent space."""

import mlx.core as mx
import mlx.nn as nn
from typing import List

from .layers import ResidualBlock, Downsample


class Encoder(nn.Module):
    """Encoder that downsamples images to latent space.

    Performs 16x spatial downsampling: 320x200 -> 20x12
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 128,
        codebook_dim: int = 256,
        num_res_blocks: int = 2,
        channel_multipliers: List[int] = [1, 2, 2, 4]
    ):
        super().__init__()

        # Initial convolution - convert NCHW to NHWC for MLX
        self.conv_in = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)

        # Downsampling blocks
        self.down_blocks = []
        channels = hidden_channels

        for i, mult in enumerate(channel_multipliers):
            out_channels = hidden_channels * mult

            # Residual blocks at current resolution
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock(channels))

            # Downsample to next level
            self.down_blocks.append(Downsample(channels, out_channels))
            channels = out_channels

        # Final residual blocks at lowest resolution
        for _ in range(num_res_blocks):
            self.down_blocks.append(ResidualBlock(channels))

        # Final layers
        self.norm_out = nn.GroupNorm(32, channels)
        self.conv_out = nn.Conv2d(channels, codebook_dim, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        """Encode image to continuous latent.

        Args:
            x: Input image (B, C, H, W) normalized to [-1, 1]

        Returns:
            z: Latent vectors (B, codebook_dim, H/16, W/16)
        """
        # Convert NCHW to NHWC for initial conv
        x = mx.transpose(x, (0, 2, 3, 1))
        x = self.conv_in(x)
        # Convert back to NCHW
        x = mx.transpose(x, (0, 3, 1, 2))

        for block in self.down_blocks:
            x = block(x)

        # Final normalization and conv - handle NCHW to NHWC
        x = mx.transpose(x, (0, 2, 3, 1))
        x = self.norm_out(x)
        x = nn.silu(x)
        x = self.conv_out(x)
        # Convert back to NCHW
        x = mx.transpose(x, (0, 3, 1, 2))

        return x
