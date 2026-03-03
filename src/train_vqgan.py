# ABOUTME: Training script for VQGAN Stage 1 with quality optimizations
# ABOUTME: Supports gradient accumulation, LR scheduling, and EMA codebook

# Fix for macOS file descriptor issue with torch multiprocessing
import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)

import argparse
import gc
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
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
    # Flatten parameters to list of (name, array) tuples
    model_flat = list(tree_flatten(model.parameters()))
    disc_flat = list(tree_flatten(discriminator.parameters()))

    # Create save dict with prefixes to avoid key collisions
    save_dict = {}
    for name, arr in model_flat:
        save_dict[f"model.{name}"] = arr
    for name, arr in disc_flat:
        save_dict[f"disc.{name}"] = arr
    save_dict["epoch"] = mx.array([epoch])

    mx.savez(path, **save_dict)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(path: str, model: VQGAN, discriminator: PatchDiscriminator):
    """Load model checkpoint and return starting epoch."""
    data = dict(mx.load(path))

    # Extract epoch
    epoch = int(data.pop("epoch").item())

    # Separate model and discriminator params by prefix
    model_params = [(k[6:], v) for k, v in data.items() if k.startswith("model.")]
    disc_params = [(k[5:], v) for k, v in data.items() if k.startswith("disc.")]

    # Load weights
    model.load_weights(model_params)
    discriminator.load_weights(disc_params)

    print(f"Loaded checkpoint from {path} (epoch {epoch})")
    return epoch


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
    """Get real images from dataset for FID computation."""
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    images = []
    for idx in indices:
        img = dataset[idx]
        img = np.array(img)
        img = ((img + 1) / 2 * 255).astype(np.uint8)
        img = img.transpose(1, 2, 0)
        images.append(img)
    return images


def generate_reconstructions(
    model: VQGAN,
    dataloader,
    num_samples: int
) -> List[np.ndarray]:
    """Generate reconstructed images for FID computation."""
    model.eval()
    images = []
    samples_collected = 0

    while samples_collected < num_samples:
        batch = next(dataloader)
        x_recon, _, _ = model(batch)

        recons = ((x_recon + 1) / 2 * 255).astype(mx.uint8)
        recons = np.array(recons)

        for i in range(len(recons)):
            if samples_collected >= num_samples:
                break
            img = recons[i].transpose(1, 2, 0)
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


def get_lr_schedule(
    base_lr: float,
    epoch: int,
    step: int,
    steps_per_epoch: int,
    warmup_epochs: int,
    total_epochs: int,
    schedule_type: str = "cosine"
) -> float:
    """Get learning rate for current step with warmup and decay."""
    global_step = epoch * steps_per_epoch + step
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    if global_step < warmup_steps:
        # Linear warmup
        return base_lr * (global_step + 1) / warmup_steps

    if schedule_type == "cosine":
        # Cosine decay
        progress = (global_step - warmup_steps) / (total_steps - warmup_steps)
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    else:
        # Constant
        return base_lr


def train_step_g(
    model: VQGAN,
    discriminator: PatchDiscriminator,
    batch: mx.array,
    disc_weight: float = 0.8,
    capture_encoder_output: bool = False
):
    """Generator training step. Returns loss and reconstructions."""
    aux_data = {}

    def g_loss_fn(model):
        # Capture encoder output before quantization if needed for dead code reset
        z = model.encoder(batch)
        if capture_encoder_output:
            aux_data['z'] = z
        z_q, _, vq_loss = model.quantizer(z)
        x_recon = model.decoder(z_q)

        aux_data['x_recon'] = x_recon
        aux_data['vq_loss'] = vq_loss

        recon_loss = reconstruction_loss(batch, x_recon)
        aux_data['recon_loss'] = recon_loss

        fake_logits = discriminator(x_recon)
        g_adv_loss = hinge_loss_g(fake_logits)

        total_loss = recon_loss + vq_loss + disc_weight * g_adv_loss
        return total_loss

    g_loss, g_grads = nn.value_and_grad(model, g_loss_fn)(model)

    result = {
        'g_loss': g_loss,
        'g_grads': g_grads,
        'x_recon': aux_data['x_recon'],
        'recon_loss': aux_data['recon_loss'],
        'vq_loss': aux_data['vq_loss'],
    }
    if capture_encoder_output:
        result['z'] = aux_data['z']
    return result


def train_step_d(
    discriminator: PatchDiscriminator,
    batch: mx.array,
    x_recon: mx.array
):
    """Discriminator training step."""
    def d_loss_fn(discriminator):
        real_logits = discriminator(batch)
        fake_logits = discriminator(mx.stop_gradient(x_recon))
        return hinge_loss_d(real_logits, fake_logits)

    d_loss, d_grads = nn.value_and_grad(discriminator, d_loss_fn)(discriminator)

    return {
        'd_loss': d_loss,
        'd_grads': d_grads,
    }


def accumulate_grads(accumulated, new_grads, accumulation_steps: int):
    """Add new gradients to accumulated gradients, scaled by accumulation steps."""
    if accumulated is None:
        # First accumulation - scale and return
        def scale(g):
            return g / accumulation_steps
        return tree_unflatten([(k, scale(v)) for k, v in tree_flatten(new_grads)])

    # Add scaled gradients
    acc_flat = dict(tree_flatten(accumulated))
    new_flat = dict(tree_flatten(new_grads))

    result = {}
    for k in acc_flat:
        result[k] = acc_flat[k] + new_flat[k] / accumulation_steps

    return tree_unflatten(list(result.items()))


def train(config_path: str, resume_path: str = None):
    """Main training loop with quality optimizations."""
    config = load_config(config_path)

    # Create output directories
    base_dir = Path(__file__).parent.parent.resolve()
    os.makedirs(f"{base_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{base_dir}/outputs/vqgan_samples", exist_ok=True)
    os.makedirs(f"{base_dir}/outputs", exist_ok=True)

    # Load dataset
    data_dir = config['data']['data_dir']
    if not os.path.isabs(data_dir):
        data_dir = f"{base_dir}/{data_dir}"
    cache_images = config['data'].get('cache_in_memory', False)
    max_images = config['data'].get('max_images', 0)
    dataset = ImageDataset(
        data_dir,
        image_size=tuple(config['data']['image_size']),
        cache_in_memory=cache_images,
        max_images=max_images
    )
    print(f"Loaded {len(dataset)} images")

    if len(dataset) == 0:
        print("ERROR: No images found. Add images to data/images/")
        return

    # Training config
    batch_size = config['training']['batch_size']
    grad_accum = config['training'].get('gradient_accumulation', 1)
    effective_batch = batch_size * grad_accum
    print(f"Batch size: {batch_size}, Gradient accumulation: {grad_accum}, Effective batch: {effective_batch}")

    dataloader = create_dataloader(dataset, batch_size=batch_size, shuffle=True)

    # Model config
    use_ema = config['training'].get('codebook_ema', False)
    ema_decay = config['training'].get('codebook_ema_decay', 0.99)
    reset_threshold = config['training'].get('codebook_reset_threshold', 2)

    model = VQGAN(
        in_channels=config['model']['in_channels'],
        hidden_channels=config['model']['hidden_channels'],
        codebook_size=config['model']['codebook_size'],
        codebook_dim=config['model']['codebook_dim'],
        num_res_blocks=config['model']['num_res_blocks'],
        use_ema=use_ema,
        ema_decay=ema_decay,
        reset_threshold=reset_threshold
    )

    discriminator = PatchDiscriminator(in_channels=config['model']['in_channels'])

    # Load checkpoint if resuming
    start_epoch = 0
    if resume_path:
        start_epoch = load_checkpoint(resume_path, model, discriminator)

    # Separate learning rates for G and D
    lr_g = config['training'].get('learning_rate_g', config['training'].get('learning_rate', 0.0001))
    lr_d = config['training'].get('learning_rate_d', lr_g * 0.2)  # Default: D is 5x slower

    optimizer_g = optim.Adam(learning_rate=lr_g)
    optimizer_d = optim.Adam(learning_rate=lr_d)

    print(f"Learning rates - G: {lr_g}, D: {lr_d}")

    # Training settings
    disc_weight = config['training'].get('disc_weight', 0.8)
    disc_train_every = config['training'].get('disc_train_every', 2)
    disc_start_epoch = config['training'].get('disc_start_epoch', 0)
    warmup_epochs = config['training'].get('lr_warmup_epochs', 5)
    lr_schedule = config['training'].get('lr_schedule', 'cosine')
    total_epochs = config['training']['num_epochs']

    print(f"Disc weight: {disc_weight}, Train D every {disc_train_every} steps, Start epoch: {disc_start_epoch}")
    print(f"LR schedule: {lr_schedule} with {warmup_epochs} warmup epochs")

    # Memory management settings (for 8GB systems)
    memory_cleanup_every = config['training'].get('memory_cleanup_every', 0)
    gc_every_epoch = config['training'].get('gc_every_epoch', False)
    if memory_cleanup_every > 0:
        print(f"Memory cleanup every {memory_cleanup_every} steps (8GB mode)")
    if gc_every_epoch:
        print("Garbage collection enabled at epoch boundaries")

    # Metrics
    metrics_config = config.get('metrics', {})
    log_file = metrics_config.get('log_file', 'outputs/training_metrics.csv')
    if not os.path.isabs(log_file):
        log_file = f"{base_dir}/{log_file}"
    fid_every = metrics_config.get('fid_every', 0)
    fid_samples = metrics_config.get('fid_samples', 500)

    logger = MetricsLogger(log_file)
    print(f"Logging metrics to {log_file}")

    # FID scorer
    fid_scorer: Optional[FIDScorer] = None
    if fid_every > 0 and is_fid_available():
        print("Initializing FID scorer...")
        fid_scorer = FIDScorer()
        real_images = get_real_images(dataset, fid_samples)
        fid_scorer.cache_real_features(real_images)
    elif fid_every > 0:
        print("FID scoring unavailable (install torch, torchvision, scipy)")

    # Training loop
    steps_per_epoch = len(dataset) // batch_size
    global_step = 0
    encoder_outputs_buffer = []  # For dead code reset

    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}")

        epoch_losses = {'g_loss': 0, 'd_loss': 0, 'recon_loss': 0, 'vq_loss': 0}
        d_steps = 0

        # Gradient accumulation buffers
        g_grads_accum = None
        d_grads_accum = None
        accum_count = 0

        for step in pbar:
            batch = next(dataloader)

            # Get current learning rates
            current_lr_g = get_lr_schedule(
                lr_g, epoch, step, steps_per_epoch,
                warmup_epochs, total_epochs, lr_schedule
            )
            current_lr_d = get_lr_schedule(
                lr_d, epoch, step, steps_per_epoch,
                warmup_epochs, total_epochs, lr_schedule
            )

            # Update optimizer learning rates
            optimizer_g.learning_rate = current_lr_g
            optimizer_d.learning_rate = current_lr_d

            # Capture encoder output every 10 steps for dead code reset (no extra forward pass)
            capture_z = (step % 10 == 0)

            # Generator forward pass
            g_result = train_step_g(model, discriminator, batch, disc_weight, capture_encoder_output=capture_z)

            # Accumulate generator gradients
            g_grads_accum = accumulate_grads(g_grads_accum, g_result['g_grads'], grad_accum)

            # Store encoder outputs for dead code reset (reuse from forward pass)
            if capture_z and 'z' in g_result:
                z = g_result['z']
                z_flat = z.transpose(0, 2, 3, 1).reshape(-1, z.shape[1])
                # Evaluate immediately to prevent graph buildup, then convert to numpy
                mx.eval(z_flat)
                encoder_outputs_buffer.append(np.array(z_flat[:32]))

                # Limit buffer size
                if len(encoder_outputs_buffer) > 20:
                    encoder_outputs_buffer = encoder_outputs_buffer[-10:]

            # Discriminator step (only if past start epoch and at right interval)
            train_d_this_step = (
                epoch >= disc_start_epoch and
                global_step % disc_train_every == 0
            )

            if train_d_this_step:
                d_result = train_step_d(discriminator, batch, g_result['x_recon'])
                d_grads_accum = accumulate_grads(d_grads_accum, d_result['d_grads'], grad_accum)
                epoch_losses['d_loss'] += float(d_result['d_loss'])
                d_steps += 1

            accum_count += 1

            # Apply accumulated gradients
            if accum_count >= grad_accum:
                # Apply generator gradients
                optimizer_g.update(model, g_grads_accum)
                mx.eval(model.parameters())

                # Apply discriminator gradients if we have any
                if d_grads_accum is not None:
                    optimizer_d.update(discriminator, d_grads_accum)
                    mx.eval(discriminator.parameters())

                # Reset accumulation and force evaluation for memory cleanup
                g_grads_accum = None
                d_grads_accum = None
                accum_count = 0

                # Force evaluation of all pending computations to free memory
                mx.eval(batch)
                mx.clear_cache()

            # Extra memory cleanup for memory-constrained systems
            if memory_cleanup_every > 0 and step % memory_cleanup_every == 0:
                # Evaluate any pending EMA state updates in the quantizer
                mx.eval(
                    model.quantizer._ema_cluster_size,
                    model.quantizer._ema_embedding_sum,
                    model.quantizer._code_usage
                )
                mx.eval(model.parameters(), discriminator.parameters())
                mx.clear_cache()
                gc.collect()

            # Track losses
            epoch_losses['g_loss'] += float(g_result['g_loss'])
            epoch_losses['recon_loss'] += float(g_result['recon_loss'])
            epoch_losses['vq_loss'] += float(g_result['vq_loss'])

            global_step += 1

            pbar.set_postfix({
                'g': f"{float(g_result['g_loss']):.3f}",
                'r': f"{float(g_result['recon_loss']):.3f}",
                'vq': f"{float(g_result['vq_loss']):.4f}",
                'lr': f"{current_lr_g:.6f}"
            })

        # Average losses
        epoch_losses['g_loss'] /= steps_per_epoch
        epoch_losses['recon_loss'] /= steps_per_epoch
        epoch_losses['vq_loss'] /= steps_per_epoch
        if d_steps > 0:
            epoch_losses['d_loss'] /= d_steps

        epoch_duration = time.time() - epoch_start
        num_samples = steps_per_epoch * batch_size

        # Codebook usage stats and dead code reset
        num_used, avg_usage = model.quantizer.get_codebook_usage()
        print(f"Epoch {epoch+1} - " +
              " | ".join([f"{k}: {v:.4f}" for k, v in epoch_losses.items()]) +
              f" | Codebook: {num_used}/512 used")

        # Reset dead codes at end of epoch
        if len(encoder_outputs_buffer) > 0:
            all_outputs = np.concatenate(encoder_outputs_buffer, axis=0)
            num_reset = model.quantizer.reset_dead_codes(mx.array(all_outputs))
            if num_reset > 0:
                print(f"Reset {num_reset} dead codebook entries")
            encoder_outputs_buffer = []

        # Garbage collection at epoch boundary (helps 8GB systems)
        if gc_every_epoch:
            mx.clear_cache()
            gc.collect()

        # Compute FID if enabled
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
            save_samples(model, sample_batch, epoch + 1, f"{base_dir}/outputs/vqgan_samples")

        # Save checkpoint
        if (epoch + 1) % config['training']['save_every'] == 0:
            save_checkpoint(
                model, discriminator, optimizer_g, optimizer_d,
                epoch + 1, f"{base_dir}/checkpoints/vqgan_epoch_{epoch+1:04d}.npz"
            )

    # Save final model
    save_checkpoint(
        model, discriminator, optimizer_g, optimizer_d,
        total_epochs, f"{base_dir}/checkpoints/vqgan_final.npz"
    )
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VQGAN")
    parser.add_argument("--config", type=str, default="configs/vqgan.yaml",
                        help="Path to config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    train(args.config, resume_path=args.resume)
