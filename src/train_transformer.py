"""Training script for Transformer Stage 2."""

import argparse
import os
from pathlib import Path
from typing import Dict, Iterator

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

from dataset import CaptionDataset
from vqgan import VQGAN
from transformer import ImageTransformer, SimpleTokenizer


def load_config(config_path: str) -> Dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_vqgan(checkpoint_path: str, config: Dict) -> VQGAN:
    """Load pretrained VQGAN."""
    model = VQGAN(
        in_channels=3,
        hidden_channels=128,
        codebook_size=config['model']['vocab_size'],
        codebook_dim=256
    )

    checkpoint = mx.load(checkpoint_path)
    model.load_weights(list(checkpoint['model'].items()))

    return model


def create_dataloader(
    dataset: CaptionDataset,
    tokenizer: SimpleTokenizer,
    vqgan: VQGAN,
    batch_size: int
) -> Iterator:
    """Create dataloader that yields (text_tokens, image_tokens) pairs."""
    indices = np.arange(len(dataset))

    while True:
        np.random.shuffle(indices)

        for start_idx in range(0, len(indices), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            if len(batch_indices) < batch_size:
                continue

            images = []
            text_tokens = []

            for idx in batch_indices:
                image, caption = dataset[idx]
                images.append(image)
                tokens = tokenizer.encode(caption, pad=True)
                text_tokens.append(tokens)

            # Stack and convert to MLX
            images = mx.array(np.stack(images))
            text_tokens = mx.array(np.array(text_tokens, dtype=np.int32))

            # Encode images to tokens
            with mx.no_grad():
                image_tokens = vqgan.encode(images)

            # Flatten spatial dims: (B, H, W) -> (B, H*W)
            B, H, W = image_tokens.shape
            image_tokens = image_tokens.reshape(B, H * W)

            yield text_tokens, image_tokens


def save_checkpoint(model: ImageTransformer, optimizer, epoch: int, path: str):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model': model.parameters(),
    }
    mx.save(path, checkpoint)
    print(f"Saved checkpoint to {path}")


def save_samples(
    model: ImageTransformer,
    vqgan: VQGAN,
    tokenizer: SimpleTokenizer,
    prompts: list,
    epoch: int,
    output_dir: str
):
    """Generate and save sample images."""
    model.eval()

    for i, prompt in enumerate(prompts):
        text_tokens = mx.array([tokenizer.encode(prompt, pad=True)])

        # Generate image tokens
        image_tokens = model.generate(text_tokens, temperature=0.9, top_k=100)

        # Reshape to spatial: (B, H*W) -> (B, H, W)
        image_tokens = image_tokens.reshape(1, 12, 20)

        # Decode to image
        with mx.no_grad():
            image = vqgan.decode(image_tokens)

        # Convert to PIL
        image = ((image[0] + 1) / 2 * 255).astype(mx.uint8)
        image = np.array(image).transpose(1, 2, 0)
        img = Image.fromarray(image)

        # Save
        safe_prompt = prompt[:50].replace(' ', '_').replace('/', '_')
        img.save(f"{output_dir}/epoch_{epoch:04d}_{i}_{safe_prompt}.png")


def train_step(
    model: ImageTransformer,
    optimizer,
    text_tokens: mx.array,
    image_tokens: mx.array
) -> Dict:
    """Single training step."""

    def loss_fn(model):
        # Shift tokens for next-token prediction
        input_tokens = image_tokens[:, :-1]
        target_tokens = image_tokens[:, 1:]

        logits = model(text_tokens, input_tokens)

        # Cross-entropy loss
        loss = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_tokens.reshape(-1),
            reduction='mean'
        )
        return loss

    loss, grads = mx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    mx.eval(model.parameters())

    return {'loss': float(loss)}


def train(config_path: str):
    """Main training loop."""
    config = load_config(config_path)

    # Create output directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs/transformer_samples", exist_ok=True)

    # Load VQGAN
    print("Loading VQGAN...")
    vqgan = load_vqgan(config['data']['vqgan_checkpoint'], config)

    # Load dataset
    dataset = CaptionDataset(
        image_dir="data/images",
        captions_file=config['data']['captions_file'],
        image_size=(320, 192)  # Divisible by 16
    )
    print(f"Loaded {len(dataset)} image-caption pairs")

    if len(dataset) == 0:
        print("ERROR: No caption data found.")
        return

    # Initialize tokenizer
    tokenizer = SimpleTokenizer(
        vocab_size=config['model'].get('text_vocab_size', 1000),
        max_length=config['model']['max_text_len']
    )

    # Create dataloader
    dataloader = create_dataloader(
        dataset, tokenizer, vqgan,
        batch_size=config['training']['batch_size']
    )

    # Initialize model
    model = ImageTransformer(
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

    # Initialize optimizer
    optimizer = optim.Adam(learning_rate=config['training']['learning_rate'])

    # Sample prompts for visualization
    sample_prompts = [
        "A dark forest with twisted trees and moonlight filtering through",
        "A medieval castle on a hilltop at sunset",
        "A cozy tavern interior with a roaring fireplace",
        "A mysterious cave entrance with glowing crystals",
    ]

    # Training loop
    steps_per_epoch = len(dataset) // config['training']['batch_size']

    for epoch in range(config['training']['num_epochs']):
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}")

        epoch_loss = 0

        for step in pbar:
            text_tokens, image_tokens = next(dataloader)
            losses = train_step(model, optimizer, text_tokens, image_tokens)

            epoch_loss += losses['loss']
            pbar.set_postfix(losses)

        avg_loss = epoch_loss / steps_per_epoch
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")

        # Save samples
        if (epoch + 1) % config['training']['sample_every'] == 0:
            save_samples(
                model, vqgan, tokenizer, sample_prompts,
                epoch + 1, "outputs/transformer_samples"
            )

        # Save checkpoint
        if (epoch + 1) % config['training']['save_every'] == 0:
            save_checkpoint(
                model, optimizer, epoch + 1,
                f"checkpoints/transformer_epoch_{epoch+1:04d}.npz"
            )

    # Save final model
    save_checkpoint(model, optimizer, config['training']['num_epochs'],
                    "checkpoints/transformer_final.npz")
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer")
    parser.add_argument("--config", type=str, default="configs/transformer.yaml",
                        help="Path to config file")
    args = parser.parse_args()

    train(args.config)
