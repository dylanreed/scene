"""Basic building blocks for VQGAN."""

import mlx.core as mx
import mlx.nn as nn


class ResidualBlock(nn.Module):
    """Residual block with two convolutions and skip connection.

    Accepts input in NCHW format (PyTorch-style) for compatibility,
    but converts to NHWC internally for MLX operations.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, channels)
        self.norm2 = nn.GroupNorm(32, channels)

    def __call__(self, x: mx.array) -> mx.array:
        # Convert NCHW to NHWC for MLX
        x = mx.transpose(x, (0, 2, 3, 1))
        residual = x

        x = self.norm1(x)
        x = nn.silu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = nn.silu(x)
        x = self.conv2(x)
        x = x + residual

        # Convert back to NCHW
        x = mx.transpose(x, (0, 3, 1, 2))
        return x


class Downsample(nn.Module):
    """Downsample spatial dimensions by 2x using strided convolution.

    Accepts input in NCHW format (PyTorch-style) for compatibility,
    but converts to NHWC internally for MLX operations.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        # Convert NCHW to NHWC for MLX
        x = mx.transpose(x, (0, 2, 3, 1))
        x = self.conv(x)
        # Convert back to NCHW
        x = mx.transpose(x, (0, 3, 1, 2))
        return x


class Upsample(nn.Module):
    """Upsample spatial dimensions by 2x using nearest neighbor + conv.

    Accepts input in NCHW format (PyTorch-style) for compatibility,
    but converts to NHWC internally for MLX operations.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        # Convert NCHW to NHWC for MLX
        x = mx.transpose(x, (0, 2, 3, 1))
        B, H, W, C = x.shape

        # Nearest neighbor upsampling
        x = mx.repeat(x, 2, axis=1)
        x = mx.repeat(x, 2, axis=2)

        x = self.conv(x)
        # Convert back to NCHW
        x = mx.transpose(x, (0, 3, 1, 2))
        return x
