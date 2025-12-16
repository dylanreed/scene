import mlx.core as mx
import sys
sys.path.insert(0, 'src')


def test_vqgan_encode_decode_roundtrip():
    """VQGAN should encode and decode images."""
    from vqgan.model import VQGAN

    model = VQGAN(
        in_channels=3,
        hidden_channels=64,
        codebook_size=256,
        codebook_dim=128
    )

    x = mx.random.normal((2, 3, 192, 320))  # Use 192 for clean divisibility

    x_recon, indices, vq_loss = model(x)

    assert x_recon.shape == x.shape
    assert indices.shape == (2, 12, 20)  # 192/16=12, 320/16=20
    assert vq_loss.shape == ()


def test_vqgan_encode_to_indices():
    """VQGAN should encode images to discrete indices."""
    from vqgan.model import VQGAN

    model = VQGAN(codebook_size=512, codebook_dim=256)
    x = mx.random.normal((1, 3, 192, 320))

    indices = model.encode(x)

    assert indices.shape == (1, 12, 20)
    assert indices.dtype == mx.int32


def test_vqgan_decode_from_indices():
    """VQGAN should decode indices back to images."""
    from vqgan.model import VQGAN

    model = VQGAN(codebook_size=512, codebook_dim=256)
    indices = mx.random.randint(0, 512, (1, 12, 20))

    x = model.decode(indices)

    assert x.shape == (1, 3, 192, 320)
