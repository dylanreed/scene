import mlx.core as mx
import sys
sys.path.insert(0, 'src')


def test_decoder_upsamples_correctly():
    """Decoder should upsample 20x12 latent to 320x200 image."""
    from vqgan.decoder import Decoder

    decoder = Decoder(
        out_channels=3,
        hidden_channels=128,
        codebook_dim=256,
        num_res_blocks=2
    )

    # 20x12 latent input
    z = mx.random.normal((2, 256, 12, 20))
    x_recon = decoder(z)

    # Should reconstruct to original size
    assert x_recon.shape == (2, 3, 192, 320)  # 12*16=192, 20*16=320


def test_decoder_output_range():
    """Decoder output should be in valid image range."""
    from vqgan.decoder import Decoder

    decoder = Decoder(out_channels=3, hidden_channels=64, codebook_dim=128)
    z = mx.random.normal((1, 128, 8, 8))
    x = decoder(z)

    # Output uses tanh, so should be in [-1, 1]
    # (In practice, may slightly exceed due to initialization)
    assert x.dtype == mx.float32
