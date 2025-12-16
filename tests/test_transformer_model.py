import mlx.core as mx
import sys
sys.path.insert(0, 'src')


def test_transformer_generates_tokens():
    """Transformer should generate image token logits."""
    from transformer.model import ImageTransformer

    model = ImageTransformer(
        vocab_size=512,      # codebook size
        text_vocab_size=1000,
        max_seq_len=240,     # 20x12 tokens
        embed_dim=256,
        num_heads=8,
        num_layers=6
    )

    # Text tokens and image tokens (for training)
    text_tokens = mx.random.randint(0, 1000, (2, 32))
    image_tokens = mx.random.randint(0, 512, (2, 240))

    logits = model(text_tokens, image_tokens)

    # Should output logits for each position
    assert logits.shape == (2, 240, 512)


def test_transformer_generates_autoregressively():
    """Transformer should generate tokens one at a time."""
    from transformer.model import ImageTransformer

    model = ImageTransformer(
        vocab_size=512,
        text_vocab_size=1000,
        max_seq_len=240,
        embed_dim=128,
        num_heads=4,
        num_layers=2
    )

    text_tokens = mx.random.randint(0, 1000, (1, 16))

    # Generate all tokens
    generated = model.generate(text_tokens, temperature=1.0)

    assert generated.shape == (1, 240)
    assert mx.all(generated >= 0)
    assert mx.all(generated < 512)
