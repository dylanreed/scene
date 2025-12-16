"""PatchGAN Discriminator for VQGAN."""

import mlx.core as mx
import mlx.nn as nn


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator that classifies image patches as real/fake.

    Outputs a spatial map of logits rather than a single value,
    which provides more gradient signal for training.

    Accepts input in NCHW format (PyTorch-style) for compatibility,
    but converts to NHWC internally for MLX operations.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        num_layers: int = 3
    ):
        super().__init__()

        layers = []

        # First layer (no normalization)
        layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1))

        # Middle layers
        channels = hidden_channels
        for i in range(1, num_layers):
            out_channels = min(channels * 2, 512)
            layers.append(nn.Conv2d(channels, out_channels, kernel_size=4, stride=2, padding=1))
            layers.append(nn.GroupNorm(32, out_channels))
            channels = out_channels

        # Final layer
        layers.append(nn.Conv2d(channels, 1, kernel_size=4, stride=1, padding=1))

        self.layers = layers

    def __call__(self, x: mx.array) -> mx.array:
        """Compute patch-wise real/fake logits.

        Args:
            x: Input image (B, C, H, W) in NCHW format

        Returns:
            logits: Patch logits (B, 1, H', W') in NCHW format
        """
        # Convert NCHW to NHWC for MLX
        x = mx.transpose(x, (0, 2, 3, 1))

        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply LeakyReLU after conv (except last layer)
            if i < len(self.layers) - 1 and isinstance(layer, nn.Conv2d):
                x = nn.leaky_relu(x, negative_slope=0.2)

        # Convert back to NCHW
        x = mx.transpose(x, (0, 3, 1, 2))
        return x
