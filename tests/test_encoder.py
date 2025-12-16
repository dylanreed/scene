import mlx.core as mx
import sys
sys.path.insert(0, 'src')


def test_encoder_downsamples_correctly():
    """Encoder should downsample 320x200 to 20x12 with correct channels."""
    from vqgan.encoder import Encoder

    encoder = Encoder(
        in_channels=3,
        hidden_channels=128,
        codebook_dim=256,
        num_res_blocks=2
    )

    # 320x200 input
    x = mx.random.normal((2, 3, 200, 320))
    z = encoder(x)

    # Should be 16x downsampled: 320/16=20, 200/16=12.5->12
    # Output channels should be codebook_dim
    assert z.shape == (2, 256, 12, 20)


def test_encoder_output_is_continuous():
    """Encoder output should be continuous (not quantized yet)."""
    from vqgan.encoder import Encoder

    encoder = Encoder(in_channels=3, hidden_channels=64, codebook_dim=128)
    x = mx.random.normal((1, 3, 64, 64))
    z = encoder(x)

    # Check it's not all integers (continuous values)
    assert z.dtype == mx.float32


if __name__ == '__main__':
    try:
        test_encoder_downsamples_correctly()
        print("✓ test_encoder_downsamples_correctly passed")
    except Exception as e:
        print(f"✗ test_encoder_downsamples_correctly failed: {e}")

    try:
        test_encoder_output_is_continuous()
        print("✓ test_encoder_output_is_continuous passed")
    except Exception as e:
        print(f"✗ test_encoder_output_is_continuous failed: {e}")
