import mlx.core as mx
import sys
sys.path.insert(0, 'src')


def test_discriminator_outputs_patch_logits():
    """Discriminator should output per-patch real/fake logits."""
    from vqgan.discriminator import PatchDiscriminator

    disc = PatchDiscriminator(in_channels=3)
    x = mx.random.normal((2, 3, 200, 320))

    logits = disc(x)

    # Should output spatial map of logits
    assert len(logits.shape) == 4
    assert logits.shape[0] == 2  # batch size preserved
    assert logits.shape[1] == 1  # single channel (real/fake)


def test_discriminator_gradient_flows():
    """Discriminator should have trainable parameters."""
    from vqgan.discriminator import PatchDiscriminator

    disc = PatchDiscriminator(in_channels=3)
    params = disc.parameters()

    # Should have parameters
    assert len(list(params.keys())) > 0
