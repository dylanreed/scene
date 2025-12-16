# Pixel Art Background Generator - Design Document

## Overview

A text-to-image model for generating VGA-era pixel art backgrounds (320x200, 256 colors) from detailed text descriptions. Trained from scratch on Apple Silicon using MLX.

## Architecture: VQGAN + Transformer

Two-stage approach optimized for pixel art's discrete nature:

### Stage 1: VQGAN (Vector Quantized GAN)

- **Encoder**: Compresses 320x200 image → 20x12 grid of discrete tokens (240 total)
- **Codebook**: Learned vocabulary of visual patterns (e.g., 512 entries)
- **Decoder**: Reconstructs image from tokens
- **Discriminator**: Ensures outputs look like real pixel art

Why this works: Pixel art has discrete colors and clear patterns. The codebook naturally learns visual primitives like "grass texture", "stone tile", "sky gradient".

### Stage 2: Transformer

- **Text Encoder**: Encodes detailed description into embeddings
- **Transformer Decoder**: Generates 240 visual tokens autoregressively
- **Cross-attention**: Links text embeddings to image token generation

### Inference Flow

```
Text Description
    → Text Encoder
    → Transformer (generates 240 tokens)
    → VQGAN Decoder
    → 320x200 pixel art image
```

## Dataset

### Sources

1. **ScummVM extractions** - Sierra SCI and LucasArts SCUMM game backgrounds
2. **Pixel art archives** - pixeljoint.com, lospec.com
3. **Screenshot databases** - MobyGames, adventure game archives

### Target Games (VGA era, ~1990-1995)

- Sierra: King's Quest V-VI, Space Quest IV-V, Gabriel Knight, Quest for Glory III-IV
- LucasArts: Monkey Island 1-2, Day of the Tentacle, Indiana Jones: Fate of Atlantis, Sam & Max
- Others: Legend of Kyrandia, Simon the Sorcerer

### Requirements

- 500-2000 background images minimum
- Detailed captions for each (generated via Claude/GPT-4V, then reviewed)
- Preprocessed to 320x200 PNG

### Structure

```
data/
  images/
    000001.png
    000002.png
    ...
  captions.jsonl  # {"image": "000001.png", "caption": "A moonlit forest..."}
```

## Training Pipeline

### Stage 1: Train VQGAN (~1-2 days on M-series Mac)

1. Load pixel art images (no captions needed)
2. Train encoder/decoder/discriminator jointly
3. Success metric: Near-perfect reconstruction of training images

Key hyperparameters:
- Codebook size: 512 (start), can try 256-1024
- Compression: 16x (320x200 → 20x12 tokens)
- Batch size: 8-16 (memory dependent)

### Stage 2: Train Transformer (~2-4 days on M-series Mac)

1. Freeze trained VQGAN
2. Encode all training images to token sequences
3. Train transformer with text conditioning
4. Success metric: Generated images match descriptions

Key hyperparameters:
- Layers: 6 (start small)
- Hidden dim: 256-512
- Heads: 8
- Context length: 240 tokens + text

## Project Structure

```
scene/
  data/
    images/              # Training images
    captions.jsonl       # Image descriptions
  src/
    vqgan/
      model.py           # VQGAN architecture
      codebook.py        # Vector quantization
      discriminator.py   # PatchGAN discriminator
    transformer/
      model.py           # Transformer architecture
      text_encoder.py    # Text encoding
    dataset.py           # Data loading utilities
    train_vqgan.py       # Stage 1 training script
    train_transformer.py # Stage 2 training script
    generate.py          # Inference/generation script
  configs/
    vqgan.yaml           # VQGAN hyperparameters
    transformer.yaml     # Transformer hyperparameters
  checkpoints/           # Saved models
  outputs/               # Generated images
  docs/
    plans/
  requirements.txt
  README.md
```

## Technology Stack

- **Framework**: MLX (Apple's ML framework for Apple Silicon)
- **Image processing**: Pillow, numpy
- **Config**: PyYAML
- **CLI**: argparse or typer

## Future Enhancements

1. **Upscaling**: Generate at higher resolutions (512x320) while maintaining pixel art style
2. **Style control**: Condition on specific game/era style
3. **Inpainting**: Edit regions of existing backgrounds
4. **Animation**: Generate parallax layers for subtle animation

## Success Criteria

1. VQGAN reconstructs training images with minimal artifacts
2. Generated images are recognizably VGA-era pixel art style
3. Generations match text descriptions (correct scene elements, mood, colors)
4. Full pipeline runs on Apple Silicon Mac with reasonable speed
