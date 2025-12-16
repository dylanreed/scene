"""Generate pixel art from text descriptions."""

import argparse
import os

import mlx.core as mx
import numpy as np
import yaml
from PIL import Image

from vqgan import VQGAN
from transformer import ImageTransformer, SimpleTokenizer


def load_models(vqgan_path: str, transformer_path: str, config_path: str):
    """Load trained models."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load VQGAN
    vqgan = VQGAN(
        in_channels=3,
        hidden_channels=128,
        codebook_size=config['model']['vocab_size'],
        codebook_dim=256
    )
    vqgan_ckpt = mx.load(vqgan_path)
    vqgan.load_weights(list(vqgan_ckpt['model'].items()))

    # Load Transformer
    transformer = ImageTransformer(
        vocab_size=config['model']['vocab_size'],
        text_vocab_size=config['model'].get('text_vocab_size', 1000),
        max_seq_len=config['model']['max_seq_len'],
        embed_dim=config['model']['embed_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        mlp_ratio=config['model']['mlp_ratio'],
        dropout=config['model']['dropout'],
        text_embed_dim=config['model']['text_embed_dim'],
        max_text_len=config['model']['max_text_len']
    )
    transformer_ckpt = mx.load(transformer_path)
    transformer.load_weights(list(transformer_ckpt['model'].items()))

    # Tokenizer
    tokenizer = SimpleTokenizer(
        vocab_size=config['model'].get('text_vocab_size', 1000),
        max_length=config['model']['max_text_len']
    )

    return vqgan, transformer, tokenizer


def generate(
    prompt: str,
    vqgan: VQGAN,
    transformer: ImageTransformer,
    tokenizer: SimpleTokenizer,
    temperature: float = 0.9,
    top_k: int = 100
) -> Image.Image:
    """Generate a pixel art image from a text prompt."""

    # Tokenize prompt
    text_tokens = mx.array([tokenizer.encode(prompt, pad=True)])

    # Generate image tokens
    image_tokens = transformer.generate(
        text_tokens,
        temperature=temperature,
        top_k=top_k
    )

    # Reshape to spatial dimensions (12x20 for 192x320)
    image_tokens = image_tokens.reshape(1, 12, 20)

    # Decode to image
    image = vqgan.decode(image_tokens)

    # Convert to PIL Image
    image = ((image[0] + 1) / 2 * 255).astype(mx.uint8)
    image = np.array(image).transpose(1, 2, 0)

    return Image.fromarray(image)


def main():
    parser = argparse.ArgumentParser(description="Generate pixel art from text")
    parser.add_argument("prompt", type=str, help="Text description of the image")
    parser.add_argument("--output", "-o", type=str, default="output.png",
                        help="Output file path")
    parser.add_argument("--vqgan", type=str, default="checkpoints/vqgan_final.npz",
                        help="VQGAN checkpoint path")
    parser.add_argument("--transformer", type=str,
                        default="checkpoints/transformer_final.npz",
                        help="Transformer checkpoint path")
    parser.add_argument("--config", type=str, default="configs/transformer.yaml",
                        help="Config file path")
    parser.add_argument("--temperature", "-t", type=float, default=0.9,
                        help="Sampling temperature (higher = more random)")
    parser.add_argument("--top-k", "-k", type=int, default=100,
                        help="Top-k sampling parameter")
    parser.add_argument("--num", "-n", type=int, default=1,
                        help="Number of images to generate")

    args = parser.parse_args()

    print("Loading models...")
    vqgan, transformer, tokenizer = load_models(
        args.vqgan, args.transformer, args.config
    )

    print(f"Generating from: \"{args.prompt}\"")

    for i in range(args.num):
        image = generate(
            args.prompt,
            vqgan,
            transformer,
            tokenizer,
            temperature=args.temperature,
            top_k=args.top_k
        )

        if args.num == 1:
            output_path = args.output
        else:
            base, ext = os.path.splitext(args.output)
            output_path = f"{base}_{i+1}{ext}"

        image.save(output_path)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
