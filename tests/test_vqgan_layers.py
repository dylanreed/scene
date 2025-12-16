import mlx.core as mx
import sys
sys.path.insert(0, 'src')


def test_residual_block_preserves_shape():
    """ResidualBlock should preserve spatial dimensions."""
    from vqgan.layers import ResidualBlock

    block = ResidualBlock(channels=64)
    x = mx.random.normal((2, 64, 32, 32))
    out = block(x)

    assert out.shape == x.shape


def test_residual_block_different_channels():
    """ResidualBlock should work with different channel counts."""
    from vqgan.layers import ResidualBlock

    for channels in [32, 64, 128, 256]:
        block = ResidualBlock(channels=channels)
        x = mx.random.normal((1, channels, 16, 16))
        out = block(x)
        assert out.shape == x.shape


def test_downsample_halves_spatial():
    """Downsample should halve spatial dimensions."""
    from vqgan.layers import Downsample

    down = Downsample(in_channels=64, out_channels=128)
    x = mx.random.normal((2, 64, 32, 32))
    out = down(x)

    assert out.shape == (2, 128, 16, 16)


def test_upsample_doubles_spatial():
    """Upsample should double spatial dimensions."""
    from vqgan.layers import Upsample

    up = Upsample(in_channels=128, out_channels=64)
    x = mx.random.normal((2, 128, 16, 16))
    out = up(x)

    assert out.shape == (2, 64, 32, 32)
