import mlx.core as mx
import sys
sys.path.insert(0, 'src')


def test_codebook_quantizes_to_discrete():
    """Codebook should map continuous vectors to discrete indices."""
    from vqgan.codebook import VectorQuantizer

    vq = VectorQuantizer(codebook_size=512, codebook_dim=256)
    z = mx.random.normal((2, 256, 20, 12))  # B, D, H, W

    z_q, indices, loss = vq(z)

    assert z_q.shape == z.shape
    assert indices.shape == (2, 20, 12)
    assert indices.dtype == mx.int32
    assert loss.shape == ()  # scalar


def test_codebook_indices_in_range():
    """Quantized indices should be in valid range."""
    from vqgan.codebook import VectorQuantizer

    vq = VectorQuantizer(codebook_size=512, codebook_dim=256)
    z = mx.random.normal((4, 256, 20, 12))

    _, indices, _ = vq(z)

    assert mx.all(indices >= 0)
    assert mx.all(indices < 512)


def test_codebook_decode_from_indices():
    """Should be able to decode indices back to vectors."""
    from vqgan.codebook import VectorQuantizer

    vq = VectorQuantizer(codebook_size=512, codebook_dim=256)
    indices = mx.random.randint(0, 512, (2, 20, 12))

    z_q = vq.decode(indices)

    assert z_q.shape == (2, 256, 20, 12)
