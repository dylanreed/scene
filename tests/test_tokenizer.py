import sys
sys.path.insert(0, 'src')


def test_tokenizer_encodes_text():
    """Tokenizer should convert text to token ids."""
    from transformer.tokenizer import SimpleTokenizer

    tokenizer = SimpleTokenizer(vocab_size=1000)
    tokens = tokenizer.encode("A forest clearing at sunset")

    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)
    assert all(0 <= t < 1000 for t in tokens)


def test_tokenizer_pads_to_length():
    """Tokenizer should pad/truncate to fixed length."""
    from transformer.tokenizer import SimpleTokenizer

    tokenizer = SimpleTokenizer(vocab_size=1000, max_length=32)

    short_text = "hello"
    long_text = "a " * 100

    short_tokens = tokenizer.encode(short_text, pad=True)
    long_tokens = tokenizer.encode(long_text, pad=True)

    assert len(short_tokens) == 32
    assert len(long_tokens) == 32


def test_tokenizer_consistent():
    """Same text should produce same tokens."""
    from transformer.tokenizer import SimpleTokenizer

    tokenizer = SimpleTokenizer(vocab_size=1000)
    text = "moonlit castle on a hill"

    tokens1 = tokenizer.encode(text)
    tokens2 = tokenizer.encode(text)

    assert tokens1 == tokens2
