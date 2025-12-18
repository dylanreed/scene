# Scene - Pixel Art Background Generator

Generate VGA-era pixel art backgrounds from text descriptions using a VQGAN + Transformer architecture, optimized for Apple Silicon.

## Architecture

1. **VQGAN**: Compresses 320x192 pixel art to 240 discrete tokens (20x12 grid)
2. **Transformer**: Generates tokens autoregressively conditioned on text

## Setup

```bash
pip install -r requirements.txt
```

## Dataset Preparation

1. Add pixel art background images (320x200 or similar) to `data/images/`
2. Create captions file `data/captions.jsonl`:
   ```json
   {"image": "forest_001.png", "caption": "A dark forest with moonlight filtering through twisted trees"}
   {"image": "castle_001.png", "caption": "A medieval castle on a hilltop at sunset with orange sky"}
   ```

## Training

### Stage 1: Train VQGAN

```bash
python src/train_vqgan.py --config configs/vqgan.yaml
```

This trains the image encoder/decoder. Watch for reconstruction quality in `outputs/vqgan_samples/`.

**What to expect:**
- Epoch 1-5: Reconstruction is noise/color soup (normal)
- Epoch 10-20: Blurry shapes, rough colors emerge
- Epoch 30-50: Recognizable structure, some detail
- Epoch 70-100: Sharp reconstructions

Samples show original (left) vs reconstruction (right). The goal is for right to match left.

### Stage 2: Train Transformer

```bash
python src/train_transformer.py --config configs/transformer.yaml
```

This trains text-to-image generation. Samples appear in `outputs/transformer_samples/`.

### Resuming Training

If training crashes or you need to stop, resume from any checkpoint:

```bash
python src/train_vqgan.py --resume checkpoints/vqgan_epoch_0050.npz
python src/train_transformer.py --resume checkpoints/transformer_epoch_0050.npz
```

Checkpoints save every epoch by default.

### Adding New Training Data

To improve the model with more images:

1. Add new images to `data/images/`
2. Update `data/captions.jsonl` with new captions
3. Resume from your best checkpoint:
   ```bash
   python src/train_vqgan.py --resume checkpoints/vqgan_epoch_0100.npz
   ```

The model trains on ALL images (old + new) but starts from existing knowledge. Usually only needs 10-20 more epochs to adapt, not another full training run.

## Generation

```bash
python src/generate.py "A spooky mansion on a cliff at night" -o mansion.png
```

Options:
- `-t, --temperature`: Sampling temperature (default: 0.9)
- `-k, --top-k`: Top-k sampling (default: 100)
- `-n, --num`: Generate multiple images

## Project Structure

```
scene/
├── configs/           # Training configurations
├── data/
│   ├── images/       # Training images
│   └── captions.jsonl
├── src/
│   ├── vqgan/        # Image encoder/decoder
│   ├── transformer/  # Text-conditioned generator
│   ├── train_vqgan.py
│   ├── train_transformer.py
│   └── generate.py
├── checkpoints/      # Saved models
└── outputs/          # Generated samples
```

## Hardware

Designed for Apple Silicon Macs using MLX.

### Recommended Specs

| RAM | batch_size | Notes |
|-----|------------|-------|
| 8GB | 4-8 | Works but slower, reduce `hidden_channels` to 64 if needed |
| 16GB | 16 | Recommended sweet spot |
| 32GB+ | 24-32 | Faster training |

### Config Tuning

Edit `configs/vqgan.yaml`:

```yaml
training:
  batch_size: 16        # increase with more RAM
  learning_rate: 0.00015  # can increase slightly with larger batches
```

### Training Tips

- **Close other apps** to free RAM (especially browsers)
- **Plug in power** - Macs throttle on battery
- **Good ventilation** - fanless Macs (Air) will thermal throttle if hot
- Watch Activity Monitor for memory pressure (yellow/red = reduce batch size)

### Training Times

- VQGAN: ~1-2 days on M1/M2/M3
- Transformer: ~2-4 days

## License

MIT
