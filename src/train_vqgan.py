"""Training script for VQGAN Stage 1."""

import argparse
import os
from pathlib import Path
from typing import Dict

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

from dataset import ImageDataset, create_dataloader
from vqgan import VQGAN, PatchDiscriminator


def load_config(config_path: str) -> Dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_checkpoint(model: VQGAN, discriminator: PatchDiscriminator,
                    optimizer_g, optimizer_d, epoch: int, path: str):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model': model.parameters(),
        'discriminator': discriminator.parameters(),
    }
    mx.save(path, checkpoint)
    print(f"Saved checkpoint to {path}")


def save_samples(model: VQGAN, batch: mx.array, epoch: int, output_dir: str):
    """Save reconstruction samples."""
    model.eval()
    x_recon, _, _ = model(batch[:4])

    # Convert to numpy and denormalize
    originals = ((batch[:4] + 1) / 2 * 255).astype(mx.uint8)
    recons = ((x_recon + 1) / 2 * 255).astype(mx.uint8)

    originals = np.array(originals)
    recons = np.array(recons)

    # Save side by side
    for i in range(min(4, len(originals))):
        orig = originals[i].transpose(1, 2, 0)
        recon = recons[i].transpose(1, 2, 0)

        combined = np.concatenate([orig, recon], axis=1)
        img = Image.fromarray(combined)
        img.save(f"{output_dir}/epoch_{epoch:04d}_sample_{i}.png")


def hinge_loss_d(real_logits: mx.array, fake_logits: mx.array) -> mx.array:
    """Hinge loss for discriminator."""
    real_loss = mx.mean(nn.relu(1.0 - real_logits))
    fake_loss = mx.mean(nn.relu(1.0 + fake_logits))
    return real_loss + fake_loss


def hinge_loss_g(fake_logits: mx.array) -> mx.array:
    """Hinge loss for generator."""
    return -mx.mean(fake_logits)


def reconstruction_loss(x: mx.array, x_recon: mx.array) -> mx.array:
    """L1 reconstruction loss."""
    return mx.mean(mx.abs(x - x_recon))


def train_step(
    model: VQGAN,
    discriminator: PatchDiscriminator,
    optimizer_g,
    optimizer_d,
    batch: mx.array,
    disc_weight: float = 0.1
):
    """Single training step."""

    # Generator forward and loss
    def g_loss_fn(model):
        x_recon, _, vq_loss = model(batch)

        # Reconstruction loss
        recon_loss = reconstruction_loss(batch, x_recon)

        # Adversarial loss
        fake_logits = discriminator(x_recon)
        g_adv_loss = hinge_loss_g(fake_logits)

        total_loss = recon_loss + vq_loss + disc_weight * g_adv_loss
        return total_loss, (recon_loss, vq_loss, g_adv_loss, x_recon)

    # Generator step
    (g_loss, aux), g_grads = mx.value_and_grad(g_loss_fn, has_aux=True)(model)
    recon_loss, vq_loss, g_adv_loss, x_recon = aux
    optimizer_g.update(model, g_grads)

    # Discriminator forward and loss
    def d_loss_fn(discriminator):
        real_logits = discriminator(batch)
        fake_logits = discriminator(mx.stop_gradient(x_recon))
        return hinge_loss_d(real_logits, fake_logits)

    # Discriminator step
    d_loss, d_grads = mx.value_and_grad(d_loss_fn)(discriminator)
    optimizer_d.update(discriminator, d_grads)

    mx.eval(model.parameters(), discriminator.parameters())

    return {
        'g_loss': float(g_loss),
        'd_loss': float(d_loss),
        'recon_loss': float(recon_loss),
        'vq_loss': float(vq_loss),
    }


def train(config_path: str):
    """Main training loop."""
    config = load_config(config_path)

    # Create output directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs/vqgan_samples", exist_ok=True)

    # Load dataset
    dataset = ImageDataset(
        config['data']['data_dir'],
        image_size=tuple(config['data']['image_size'])
    )
    print(f"Loaded {len(dataset)} images")

    if len(dataset) == 0:
        print("ERROR: No images found. Add images to data/images/")
        return

    dataloader = create_dataloader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )

    # Initialize models
    model = VQGAN(
        in_channels=config['model']['in_channels'],
        hidden_channels=config['model']['hidden_channels'],
        codebook_size=config['model']['codebook_size'],
        codebook_dim=config['model']['codebook_dim'],
        num_res_blocks=config['model']['num_res_blocks']
    )

    discriminator = PatchDiscriminator(in_channels=config['model']['in_channels'])

    # Initialize optimizers
    optimizer_g = optim.Adam(learning_rate=config['training']['learning_rate'])
    optimizer_d = optim.Adam(learning_rate=config['training']['learning_rate'])

    # Training loop
    steps_per_epoch = len(dataset) // config['training']['batch_size']

    for epoch in range(config['training']['num_epochs']):
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}")

        epoch_losses = {'g_loss': 0, 'd_loss': 0, 'recon_loss': 0, 'vq_loss': 0}

        for step in pbar:
            batch = next(dataloader)
            losses = train_step(model, discriminator, optimizer_g, optimizer_d, batch)

            for k, v in losses.items():
                epoch_losses[k] += v

            pbar.set_postfix({k: f"{v:.4f}" for k, v in losses.items()})

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= steps_per_epoch

        print(f"Epoch {epoch+1} - " + " | ".join([f"{k}: {v:.4f}" for k, v in epoch_losses.items()]))

        # Save samples
        if (epoch + 1) % config['training']['sample_every'] == 0:
            sample_batch = next(dataloader)
            save_samples(model, sample_batch, epoch + 1, "outputs/vqgan_samples")

        # Save checkpoint
        if (epoch + 1) % config['training']['save_every'] == 0:
            save_checkpoint(
                model, discriminator, optimizer_g, optimizer_d,
                epoch + 1, f"checkpoints/vqgan_epoch_{epoch+1:04d}.npz"
            )

    # Save final model
    save_checkpoint(
        model, discriminator, optimizer_g, optimizer_d,
        config['training']['num_epochs'], "checkpoints/vqgan_final.npz"
    )
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VQGAN")
    parser.add_argument("--config", type=str, default="configs/vqgan.yaml",
                        help="Path to config file")
    args = parser.parse_args()

    train(args.config)
