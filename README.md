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

### Stage 2: Train Transformer

```bash
python src/train_transformer.py --config configs/transformer.yaml
```

This trains text-to-image generation. Samples appear in `outputs/transformer_samples/`.

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

Designed for Apple Silicon Macs using MLX. Training times:
- VQGAN: ~1-2 days on M1/M2/M3
- Transformer: ~2-4 days

## License

MIT
