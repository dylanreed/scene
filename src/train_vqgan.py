"""Training script for VQGAN Stage 1."""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

from dataset import ImageDataset, create_dataloader
from vqgan import VQGAN, PatchDiscriminator
from metrics import MetricsLogger, FIDScorer, is_fid_available


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


def load_checkpoint(path: str, model: VQGAN, discriminator: PatchDiscriminator):
    """Load model checkpoint and return starting epoch."""
    checkpoint = mx.load(path)

    # Load model weights
    model.load_weights(list(checkpoint['model'].items()))
    discriminator.load_weights(list(checkpoint['discriminator'].items()))

    start_epoch = checkpoint['epoch']
    print(f"Loaded checkpoint from {path} (epoch {start_epoch})")
    return start_epoch


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


def get_real_images(dataset: ImageDataset, num_samples: int) -> List[np.ndarray]:
    """Get real images from dataset for FID computation.

    Args:
        dataset: Image dataset
        num_samples: Number of images to sample

    Returns:
        List of numpy arrays (H, W, 3) in range [0, 255]
    """
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    images = []
    for idx in indices:
        # Dataset returns normalized image, denormalize
        img = dataset[idx]  # (C, H, W) in [-1, 1]
        img = np.array(img)
        img = ((img + 1) / 2 * 255).astype(np.uint8)
        img = img.transpose(1, 2, 0)  # (H, W, C)
        images.append(img)
    return images


def generate_reconstructions(
    model: VQGAN,
    dataloader,
    num_samples: int
) -> List[np.ndarray]:
    """Generate reconstructed images for FID computation.

    Args:
        model: VQGAN model
        dataloader: Data loader
        num_samples: Number of reconstructions to generate

    Returns:
        List of numpy arrays (H, W, 3) in range [0, 255]
    """
    model.eval()
    images = []
    samples_collected = 0

    while samples_collected < num_samples:
        batch = next(dataloader)
        x_recon, _, _ = model(batch)

        # Denormalize
        recons = ((x_recon + 1) / 2 * 255).astype(mx.uint8)
        recons = np.array(recons)

        for i in range(len(recons)):
            if samples_collected >= num_samples:
                break
            img = recons[i].transpose(1, 2, 0)  # (H, W, C)
            images.append(img)
            samples_collected += 1

    return images


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

    # Store auxiliary values
    aux_data = {}

    # Generator forward and loss
    def g_loss_fn(model):
        x_recon, _, vq_loss = model(batch)
        aux_data['x_recon'] = x_recon
        aux_data['vq_loss'] = vq_loss

        # Reconstruction loss
        recon_loss = reconstruction_loss(batch, x_recon)
        aux_data['recon_loss'] = recon_loss

        # Adversarial loss
        fake_logits = discriminator(x_recon)
        g_adv_loss = hinge_loss_g(fake_logits)

        total_loss = recon_loss + vq_loss + disc_weight * g_adv_loss
        return total_loss

    # Generator step
    g_loss, g_grads = nn.value_and_grad(model, g_loss_fn)(model)
    optimizer_g.update(model, g_grads)

    x_recon = aux_data['x_recon']
    recon_loss = aux_data['recon_loss']
    vq_loss = aux_data['vq_loss']

    # Discriminator forward and loss
    def d_loss_fn(discriminator):
        real_logits = discriminator(batch)
        fake_logits = discriminator(mx.stop_gradient(x_recon))
        return hinge_loss_d(real_logits, fake_logits)

    # Discriminator step
    d_loss, d_grads = nn.value_and_grad(discriminator, d_loss_fn)(discriminator)
    optimizer_d.update(discriminator, d_grads)

    mx.eval(model.parameters(), discriminator.parameters())

    return {
        'g_loss': float(g_loss),
        'd_loss': float(d_loss),
        'recon_loss': float(recon_loss),
        'vq_loss': float(vq_loss),
    }


def train(config_path: str, resume_path: str = None):
    """Main training loop."""
    config = load_config(config_path)

    # Create output directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs/vqgan_samples", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

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

    # Load checkpoint if resuming
    start_epoch = 0
    if resume_path:
        start_epoch = load_checkpoint(resume_path, model, discriminator)

    # Initialize optimizers
    optimizer_g = optim.Adam(learning_rate=config['training']['learning_rate'])
    optimizer_d = optim.Adam(learning_rate=config['training']['learning_rate'])

    # Initialize metrics
    metrics_config = config.get('metrics', {})
    log_file = metrics_config.get('log_file', 'outputs/training_metrics.csv')
    fid_every = metrics_config.get('fid_every', 5)
    fid_samples = metrics_config.get('fid_samples', 500)
    fid_full_at_end = metrics_config.get('fid_full_at_end', True)

    logger = MetricsLogger(log_file)
    print(f"Logging metrics to {log_file}")

    # Initialize FID scorer if available
    fid_scorer: Optional[FIDScorer] = None
    if fid_every > 0 and is_fid_available():
        print("Initializing FID scorer...")
        fid_scorer = FIDScorer()
        real_images = get_real_images(dataset, fid_samples)
        fid_scorer.cache_real_features(real_images)
    elif fid_every > 0:
        print("FID scoring unavailable (install torch, torchvision, scipy)")

    # Training loop
    steps_per_epoch = len(dataset) // config['training']['batch_size']
    total_epochs = config['training']['num_epochs']
    batch_size = config['training']['batch_size']

    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()
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

        epoch_duration = time.time() - epoch_start
        num_samples = steps_per_epoch * batch_size

        print(f"Epoch {epoch+1} - " + " | ".join([f"{k}: {v:.4f}" for k, v in epoch_losses.items()]))

        # Compute FID if enabled and this is an FID epoch
        fid_score = None
        if fid_scorer and fid_every > 0 and (epoch + 1) % fid_every == 0:
            print("Computing FID...")
            recon_images = generate_reconstructions(model, dataloader, fid_samples)
            fid_score = fid_scorer.compute_fid(recon_images)
            print(f"FID: {fid_score:.4f}")

        # Log metrics
        logger.log_epoch(
            epoch=epoch + 1,
            losses=epoch_losses,
            duration=epoch_duration,
            num_samples=num_samples,
            fid=fid_score
        )

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

    # Compute final FID on full dataset if enabled
    if fid_scorer and fid_full_at_end:
        print("Computing final FID on full dataset...")
        full_real_images = get_real_images(dataset, len(dataset))
        fid_scorer.cache_real_features(full_real_images)
        full_recon_images = generate_reconstructions(model, dataloader, len(dataset))
        final_fid = fid_scorer.compute_fid(full_recon_images)
        print(f"Final FID (full dataset): {final_fid:.4f}")

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
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (e.g., checkpoints/vqgan_epoch_0010.npz)")
    args = parser.parse_args()

    train(args.config, resume_path=args.resume)
