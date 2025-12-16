import mlx.core as mx
import sys
sys.path.insert(0, 'src')


def test_text_encoder_embeds_tokens():
    """Text encoder should convert tokens to embeddings."""
    from transformer.text_encoder import TextEncoder

    encoder = TextEncoder(vocab_size=1000, embed_dim=256, max_length=128)

    # Batch of 2, sequence length 32
    tokens = mx.random.randint(0, 1000, (2, 32))
    embeddings = encoder(tokens)

    assert embeddings.shape == (2, 32, 256)


def test_text_encoder_with_mask():
    """Text encoder should handle attention masks."""
    from transformer.text_encoder import TextEncoder

    encoder = TextEncoder(vocab_size=1000, embed_dim=256)
    tokens = mx.random.randint(0, 1000, (1, 16))

    embeddings = encoder(tokens)

    assert embeddings.shape == (1, 16, 256)
    assert embeddings.dtype == mx.float32
