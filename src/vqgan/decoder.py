"""VQGAN Decoder - reconstructs images from quantized latents."""

import mlx.core as mx
import mlx.nn as nn
from typing import List

from .layers import ResidualBlock, Upsample


class Decoder(nn.Module):
    """Decoder that upsamples latents back to images.

    Performs 16x spatial upsampling: 20x12 -> 320x192
    (Note: 200 doesn't divide evenly by 16, so we get 192)
    """

    def __init__(
        self,
        out_channels: int = 3,
        hidden_channels: int = 128,
        codebook_dim: int = 256,
        num_res_blocks: int = 2,
        channel_multipliers: List[int] = [1, 2, 2, 4]
    ):
        super().__init__()

        # Reverse multipliers for upsampling
        channel_multipliers = list(reversed(channel_multipliers))
        initial_channels = hidden_channels * channel_multipliers[0]

        # Initial convolution from codebook dim
        self.conv_in = nn.Conv2d(codebook_dim, initial_channels, kernel_size=3, padding=1)

        # Upsampling blocks
        self.up_blocks = []
        channels = initial_channels

        # First set of residual blocks at lowest resolution
        for _ in range(num_res_blocks):
            self.up_blocks.append(ResidualBlock(channels))

        # Upsample through each level
        for i, mult in enumerate(channel_multipliers):
            # Determine next channel count
            if i < len(channel_multipliers) - 1:
                next_mult = channel_multipliers[i + 1]
                next_ch = hidden_channels * next_mult
            else:
                next_ch = hidden_channels  # Last level goes back to hidden_channels

            # Upsample to next level
            self.up_blocks.append(Upsample(channels, next_ch))
            channels = next_ch

            # Residual blocks at this resolution
            for _ in range(num_res_blocks):
                self.up_blocks.append(ResidualBlock(channels))

        # Final layers
        self.norm_out = nn.GroupNorm(32, channels)
        self.conv_out = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)

    def __call__(self, z: mx.array) -> mx.array:
        """Decode latent to image.

        Args:
            z: Latent vectors (B, codebook_dim, H, W)

        Returns:
            x: Reconstructed image (B, C, H*16, W*16) in [-1, 1]
        """
        # Convert NCHW to NHWC for initial conv
        x = mx.transpose(z, (0, 2, 3, 1))
        x = self.conv_in(x)
        # Convert back to NCHW
        x = mx.transpose(x, (0, 3, 1, 2))

        for block in self.up_blocks:
            x = block(x)

        # Final normalization and conv - handle NCHW to NHWC
        x = mx.transpose(x, (0, 2, 3, 1))
        x = self.norm_out(x)
        x = nn.silu(x)
        x = self.conv_out(x)
        # Convert back to NCHW
        x = mx.transpose(x, (0, 3, 1, 2))
        x = mx.tanh(x)

        return x
