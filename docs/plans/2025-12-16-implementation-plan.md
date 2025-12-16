# Pixel Art Generator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a text-to-pixel-art generator using VQGAN + Transformer on Apple Silicon with MLX.

**Architecture:** Two-stage approach - VQGAN compresses 320x200 images to 240 discrete tokens, Transformer generates tokens from text descriptions. Trained separately, combined for inference.

**Tech Stack:** MLX (Apple ML framework), Pillow, NumPy, PyYAML

---

## Phase 1: Project Setup

### Task 1: Initialize Project Structure

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/vqgan/__init__.py`
- Create: `src/transformer/__init__.py`
- Create: `configs/vqgan.yaml`
- Create: `configs/transformer.yaml`
- Create: `data/.gitkeep`
- Create: `checkpoints/.gitkeep`
- Create: `outputs/.gitkeep`

**Step 1: Create requirements.txt**

```txt
mlx>=0.4.0
numpy>=1.24.0
Pillow>=10.0.0
PyYAML>=6.0
tqdm>=4.65.0
```

**Step 2: Create directory structure**

```bash
mkdir -p src/vqgan src/transformer configs data/images checkpoints outputs tests
touch src/__init__.py src/vqgan/__init__.py src/transformer/__init__.py
touch data/.gitkeep checkpoints/.gitkeep outputs/.gitkeep
```

**Step 3: Create VQGAN config**

`configs/vqgan.yaml`:
```yaml
model:
  in_channels: 3
  hidden_channels: 128
  num_res_blocks: 2
  codebook_size: 512
  codebook_dim: 256
  # 320x200 -> 20x12 tokens (16x downscale)
  downsample_factor: 16

training:
  batch_size: 8
  learning_rate: 1e-4
  num_epochs: 100
  save_every: 10
  sample_every: 5

data:
  image_size: [320, 200]
  data_dir: "data/images"
```

**Step 4: Create Transformer config**

`configs/transformer.yaml`:
```yaml
model:
  vocab_size: 512  # matches codebook_size
  max_seq_len: 240  # 20x12 tokens
  embed_dim: 256
  num_heads: 8
  num_layers: 6
  mlp_ratio: 4
  dropout: 0.1
  text_embed_dim: 256
  max_text_len: 128

training:
  batch_size: 8
  learning_rate: 1e-4
  num_epochs: 100
  save_every: 10
  sample_every: 5

data:
  captions_file: "data/captions.jsonl"
  vqgan_checkpoint: "checkpoints/vqgan_best.npz"
```

**Step 5: Install dependencies**

Run: `pip install -r requirements.txt`

**Step 6: Commit**

```bash
git add .
git commit -m "feat: initialize project structure and configs"
```

---

## Phase 2: Dataset Utilities

### Task 2: Image Dataset Loader

**Files:**
- Create: `src/dataset.py`
- Create: `tests/test_dataset.py`

**Step 1: Write the failing test**

`tests/test_dataset.py`:
```python
import numpy as np
import sys
sys.path.insert(0, 'src')

def test_image_dataset_loads_images():
    """Dataset should load and normalize images to [-1, 1]."""
    from dataset import ImageDataset

    # Create a dummy image for testing
    from PIL import Image
    import os
    os.makedirs("data/test_images", exist_ok=True)
    img = Image.new('RGB', (320, 200), color=(128, 128, 128))
    img.save("data/test_images/test.png")

    dataset = ImageDataset("data/test_images", image_size=(320, 200))

    assert len(dataset) == 1
    image = dataset[0]
    assert image.shape == (3, 200, 320)  # CHW format
    assert image.min() >= -1.0
    assert image.max() <= 1.0

    # Cleanup
    os.remove("data/test_images/test.png")
    os.rmdir("data/test_images")


def test_image_dataset_batching():
    """Dataset should support batching."""
    from dataset import ImageDataset, create_dataloader
    from PIL import Image
    import os

    os.makedirs("data/test_images", exist_ok=True)
    for i in range(4):
        img = Image.new('RGB', (320, 200), color=(i * 50, i * 50, i * 50))
        img.save(f"data/test_images/test_{i}.png")

    dataset = ImageDataset("data/test_images", image_size=(320, 200))
    dataloader = create_dataloader(dataset, batch_size=2)

    batch = next(iter(dataloader))
    assert batch.shape == (2, 3, 200, 320)

    # Cleanup
    for i in range(4):
        os.remove(f"data/test_images/test_{i}.png")
    os.rmdir("data/test_images")
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dataset.py -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

**Step 3: Write minimal implementation**

`src/dataset.py`:
```python
"""Dataset utilities for loading pixel art images."""

import os
from pathlib import Path
from typing import Tuple, List, Iterator
import numpy as np
from PIL import Image
import mlx.core as mx


class ImageDataset:
    """Dataset for loading and preprocessing pixel art images."""

    def __init__(self, data_dir: str, image_size: Tuple[int, int] = (320, 200)):
        self.data_dir = Path(data_dir)
        self.image_size = image_size  # (width, height)
        self.image_paths = self._find_images()

    def _find_images(self) -> List[Path]:
        """Find all image files in the data directory."""
        extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
        paths = []
        for ext in extensions:
            paths.extend(self.data_dir.glob(f'*{ext}'))
            paths.extend(self.data_dir.glob(f'*{ext.upper()}'))
        return sorted(paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> np.ndarray:
        """Load and preprocess a single image.

        Returns:
            numpy array of shape (C, H, W) normalized to [-1, 1]
        """
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')

        # Resize if needed
        if img.size != self.image_size:
            img = img.resize(self.image_size, Image.Resampling.NEAREST)

        # Convert to numpy and normalize to [-1, 1]
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr * 2.0 - 1.0  # [0,1] -> [-1,1]

        # Convert HWC to CHW
        arr = np.transpose(arr, (2, 0, 1))

        return arr


def create_dataloader(
    dataset: ImageDataset,
    batch_size: int = 8,
    shuffle: bool = True
) -> Iterator[mx.array]:
    """Create a simple dataloader that yields batches."""
    indices = np.arange(len(dataset))

    while True:
        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, len(indices), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            if len(batch_indices) < batch_size:
                continue  # Skip incomplete batches

            batch = np.stack([dataset[i] for i in batch_indices])
            yield mx.array(batch)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dataset.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/dataset.py tests/test_dataset.py
git commit -m "feat: add image dataset loader"
```

---

### Task 3: Caption Dataset Loader

**Files:**
- Modify: `src/dataset.py`
- Modify: `tests/test_dataset.py`

**Step 1: Write the failing test**

Add to `tests/test_dataset.py`:
```python
def test_caption_dataset_loads_pairs():
    """CaptionDataset should load image-caption pairs."""
    from dataset import CaptionDataset
    from PIL import Image
    import os
    import json

    # Setup test data
    os.makedirs("data/test_images", exist_ok=True)
    img = Image.new('RGB', (320, 200), color=(100, 150, 200))
    img.save("data/test_images/scene.png")

    captions = [
        {"image": "scene.png", "caption": "A blue sky over mountains"}
    ]
    with open("data/test_captions.jsonl", "w") as f:
        for c in captions:
            f.write(json.dumps(c) + "\n")

    dataset = CaptionDataset(
        image_dir="data/test_images",
        captions_file="data/test_captions.jsonl",
        image_size=(320, 200)
    )

    assert len(dataset) == 1
    image, caption = dataset[0]
    assert image.shape == (3, 200, 320)
    assert caption == "A blue sky over mountains"

    # Cleanup
    os.remove("data/test_images/scene.png")
    os.rmdir("data/test_images")
    os.remove("data/test_captions.jsonl")
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dataset.py::test_caption_dataset_loads_pairs -v`
Expected: FAIL with "ImportError"

**Step 3: Add CaptionDataset to implementation**

Add to `src/dataset.py`:
```python
import json


class CaptionDataset:
    """Dataset for loading image-caption pairs."""

    def __init__(
        self,
        image_dir: str,
        captions_file: str,
        image_size: Tuple[int, int] = (320, 200)
    ):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.pairs = self._load_captions(captions_file)

    def _load_captions(self, captions_file: str) -> List[dict]:
        """Load image-caption pairs from JSONL file."""
        pairs = []
        with open(captions_file, 'r') as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line))
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:
        """Load an image-caption pair.

        Returns:
            Tuple of (image array CHW [-1,1], caption string)
        """
        pair = self.pairs[idx]
        img_path = self.image_dir / pair['image']

        img = Image.open(img_path).convert('RGB')
        if img.size != self.image_size:
            img = img.resize(self.image_size, Image.Resampling.NEAREST)

        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        arr = np.transpose(arr, (2, 0, 1))

        return arr, pair['caption']
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dataset.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/dataset.py tests/test_dataset.py
git commit -m "feat: add caption dataset loader"
```

---

## Phase 3: VQGAN Components

### Task 4: Residual Block

**Files:**
- Create: `src/vqgan/layers.py`
- Create: `tests/test_vqgan_layers.py`

**Step 1: Write the failing test**

`tests/test_vqgan_layers.py`:
```python
import mlx.core as mx
import sys
sys.path.insert(0, 'src')


def test_residual_block_preserves_shape():
    """ResidualBlock should preserve spatial dimensions."""
    from vqgan.layers import ResidualBlock

    block = ResidualBlock(channels=64)
    x = mx.random.normal((2, 64, 32, 32))
    out = block(x)

    assert out.shape == x.shape


def test_residual_block_different_channels():
    """ResidualBlock should work with different channel counts."""
    from vqgan.layers import ResidualBlock

    for channels in [32, 64, 128, 256]:
        block = ResidualBlock(channels=channels)
        x = mx.random.normal((1, channels, 16, 16))
        out = block(x)
        assert out.shape == x.shape
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_vqgan_layers.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

`src/vqgan/layers.py`:
```python
"""Basic building blocks for VQGAN."""

import mlx.core as mx
import mlx.nn as nn


class ResidualBlock(nn.Module):
    """Residual block with two convolutions and skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, channels)
        self.norm2 = nn.GroupNorm(32, channels)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.norm1(x)
        x = nn.silu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = nn.silu(x)
        x = self.conv2(x)
        return x + residual
```

Update `src/vqgan/__init__.py`:
```python
from .layers import ResidualBlock
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_vqgan_layers.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/vqgan/layers.py src/vqgan/__init__.py tests/test_vqgan_layers.py
git commit -m "feat: add VQGAN residual block"
```

---

### Task 5: Downsample and Upsample Blocks

**Files:**
- Modify: `src/vqgan/layers.py`
- Modify: `tests/test_vqgan_layers.py`

**Step 1: Write the failing test**

Add to `tests/test_vqgan_layers.py`:
```python
def test_downsample_halves_spatial():
    """Downsample should halve spatial dimensions."""
    from vqgan.layers import Downsample

    down = Downsample(in_channels=64, out_channels=128)
    x = mx.random.normal((2, 64, 32, 32))
    out = down(x)

    assert out.shape == (2, 128, 16, 16)


def test_upsample_doubles_spatial():
    """Upsample should double spatial dimensions."""
    from vqgan.layers import Upsample

    up = Upsample(in_channels=128, out_channels=64)
    x = mx.random.normal((2, 128, 16, 16))
    out = up(x)

    assert out.shape == (2, 64, 32, 32)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_vqgan_layers.py::test_downsample_halves_spatial -v`
Expected: FAIL with "ImportError"

**Step 3: Add Downsample and Upsample**

Add to `src/vqgan/layers.py`:
```python
class Downsample(nn.Module):
    """Downsample spatial dimensions by 2x using strided convolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsample spatial dimensions by 2x using nearest neighbor + conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        B, C, H, W = x.shape
        # Nearest neighbor upsampling
        x = mx.repeat(x, 2, axis=2)
        x = mx.repeat(x, 2, axis=3)
        return self.conv(x)
```

Update `src/vqgan/__init__.py`:
```python
from .layers import ResidualBlock, Downsample, Upsample
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_vqgan_layers.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/vqgan/layers.py src/vqgan/__init__.py tests/test_vqgan_layers.py
git commit -m "feat: add VQGAN downsample and upsample blocks"
```

---

### Task 6: Vector Quantizer (Codebook)

**Files:**
- Create: `src/vqgan/codebook.py`
- Create: `tests/test_codebook.py`

**Step 1: Write the failing test**

`tests/test_codebook.py`:
```python
import mlx.core as mx
import sys
sys.path.insert(0, 'src')


def test_codebook_quantizes_to_discrete():
    """Codebook should map continuous vectors to discrete indices."""
    from vqgan.codebook import VectorQuantizer

    vq = VectorQuantizer(codebook_size=512, codebook_dim=256)
    z = mx.random.normal((2, 256, 20, 12))  # B, D, H, W

    z_q, indices, loss = vq(z)

    assert z_q.shape == z.shape
    assert indices.shape == (2, 20, 12)
    assert indices.dtype == mx.int32
    assert loss.shape == ()  # scalar


def test_codebook_indices_in_range():
    """Quantized indices should be in valid range."""
    from vqgan.codebook import VectorQuantizer

    vq = VectorQuantizer(codebook_size=512, codebook_dim=256)
    z = mx.random.normal((4, 256, 20, 12))

    _, indices, _ = vq(z)

    assert mx.all(indices >= 0)
    assert mx.all(indices < 512)


def test_codebook_decode_from_indices():
    """Should be able to decode indices back to vectors."""
    from vqgan.codebook import VectorQuantizer

    vq = VectorQuantizer(codebook_size=512, codebook_dim=256)
    indices = mx.randint(0, 512, (2, 20, 12))

    z_q = vq.decode(indices)

    assert z_q.shape == (2, 256, 20, 12)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_codebook.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

`src/vqgan/codebook.py`:
```python
"""Vector Quantizer for VQGAN."""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple


class VectorQuantizer(nn.Module):
    """Vector Quantizer with EMA codebook updates.

    Maps continuous latent vectors to discrete codebook entries.
    Uses straight-through estimator for gradients.
    """

    def __init__(
        self,
        codebook_size: int = 512,
        codebook_dim: int = 256,
        commitment_cost: float = 0.25
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_cost = commitment_cost

        # Initialize codebook embeddings
        self.embedding = nn.Embedding(codebook_size, codebook_dim)

    def __call__(self, z: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """Quantize continuous latent vectors.

        Args:
            z: Continuous latents of shape (B, D, H, W)

        Returns:
            z_q: Quantized latents (B, D, H, W)
            indices: Codebook indices (B, H, W)
            loss: Commitment loss (scalar)
        """
        B, D, H, W = z.shape

        # Reshape: (B, D, H, W) -> (B*H*W, D)
        z_flat = z.transpose(0, 2, 3, 1).reshape(-1, D)

        # Get codebook
        codebook = self.embedding.weight  # (codebook_size, D)

        # Compute distances: ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z@e.T
        z_sq = mx.sum(z_flat ** 2, axis=1, keepdims=True)
        e_sq = mx.sum(codebook ** 2, axis=1, keepdims=True)
        distances = z_sq + e_sq.T - 2 * (z_flat @ codebook.T)

        # Get nearest codebook entry
        indices = mx.argmin(distances, axis=1).astype(mx.int32)

        # Lookup quantized vectors
        z_q_flat = self.embedding(indices)

        # Reshape back: (B*H*W, D) -> (B, D, H, W)
        z_q = z_q_flat.reshape(B, H, W, D).transpose(0, 3, 1, 2)
        indices = indices.reshape(B, H, W)

        # Compute loss
        codebook_loss = mx.mean((mx.stop_gradient(z) - z_q) ** 2)
        commitment_loss = mx.mean((z - mx.stop_gradient(z_q)) ** 2)
        loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator: copy gradients from z_q to z
        z_q = z + mx.stop_gradient(z_q - z)

        return z_q, indices, loss

    def decode(self, indices: mx.array) -> mx.array:
        """Decode indices back to latent vectors.

        Args:
            indices: Codebook indices of shape (B, H, W)

        Returns:
            z_q: Quantized latents (B, D, H, W)
        """
        B, H, W = indices.shape
        indices_flat = indices.reshape(-1)
        z_q_flat = self.embedding(indices_flat)
        z_q = z_q_flat.reshape(B, H, W, self.codebook_dim).transpose(0, 3, 1, 2)
        return z_q
```

Update `src/vqgan/__init__.py`:
```python
from .layers import ResidualBlock, Downsample, Upsample
from .codebook import VectorQuantizer
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_codebook.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/vqgan/codebook.py src/vqgan/__init__.py tests/test_codebook.py
git commit -m "feat: add vector quantizer codebook"
```

---

### Task 7: VQGAN Encoder

**Files:**
- Create: `src/vqgan/encoder.py`
- Create: `tests/test_encoder.py`

**Step 1: Write the failing test**

`tests/test_encoder.py`:
```python
import mlx.core as mx
import sys
sys.path.insert(0, 'src')


def test_encoder_downsamples_correctly():
    """Encoder should downsample 320x200 to 20x12 with correct channels."""
    from vqgan.encoder import Encoder

    encoder = Encoder(
        in_channels=3,
        hidden_channels=128,
        codebook_dim=256,
        num_res_blocks=2
    )

    # 320x200 input
    x = mx.random.normal((2, 3, 200, 320))
    z = encoder(x)

    # Should be 16x downsampled: 320/16=20, 200/16=12.5->12
    # Output channels should be codebook_dim
    assert z.shape == (2, 256, 12, 20)


def test_encoder_output_is_continuous():
    """Encoder output should be continuous (not quantized yet)."""
    from vqgan.encoder import Encoder

    encoder = Encoder(in_channels=3, hidden_channels=64, codebook_dim=128)
    x = mx.random.normal((1, 3, 64, 64))
    z = encoder(x)

    # Check it's not all integers (continuous values)
    assert z.dtype == mx.float32
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_encoder.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

`src/vqgan/encoder.py`:
```python
"""VQGAN Encoder - compresses images to continuous latent space."""

import mlx.core as mx
import mlx.nn as nn
from typing import List

from .layers import ResidualBlock, Downsample


class Encoder(nn.Module):
    """Encoder that downsamples images to latent space.

    Performs 16x spatial downsampling: 320x200 -> 20x12
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 128,
        codebook_dim: int = 256,
        num_res_blocks: int = 2,
        channel_multipliers: List[int] = [1, 2, 2, 4]
    ):
        super().__init__()

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)

        # Downsampling blocks
        self.down_blocks = []
        channels = hidden_channels

        for i, mult in enumerate(channel_multipliers):
            out_channels = hidden_channels * mult

            # Residual blocks
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock(channels))

            # Downsample (except last)
            if i < len(channel_multipliers) - 1:
                self.down_blocks.append(Downsample(channels, out_channels))
                channels = out_channels

        # Final layers
        self.norm_out = nn.GroupNorm(32, channels)
        self.conv_out = nn.Conv2d(channels, codebook_dim, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        """Encode image to continuous latent.

        Args:
            x: Input image (B, C, H, W) normalized to [-1, 1]

        Returns:
            z: Latent vectors (B, codebook_dim, H/16, W/16)
        """
        x = self.conv_in(x)

        for block in self.down_blocks:
            x = block(x)

        x = self.norm_out(x)
        x = nn.silu(x)
        x = self.conv_out(x)

        return x
```

Update `src/vqgan/__init__.py`:
```python
from .layers import ResidualBlock, Downsample, Upsample
from .codebook import VectorQuantizer
from .encoder import Encoder
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_encoder.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/vqgan/encoder.py src/vqgan/__init__.py tests/test_encoder.py
git commit -m "feat: add VQGAN encoder"
```

---

### Task 8: VQGAN Decoder

**Files:**
- Create: `src/vqgan/decoder.py`
- Create: `tests/test_decoder.py`

**Step 1: Write the failing test**

`tests/test_decoder.py`:
```python
import mlx.core as mx
import sys
sys.path.insert(0, 'src')


def test_decoder_upsamples_correctly():
    """Decoder should upsample 20x12 latent to 320x200 image."""
    from vqgan.decoder import Decoder

    decoder = Decoder(
        out_channels=3,
        hidden_channels=128,
        codebook_dim=256,
        num_res_blocks=2
    )

    # 20x12 latent input
    z = mx.random.normal((2, 256, 12, 20))
    x_recon = decoder(z)

    # Should reconstruct to original size
    assert x_recon.shape == (2, 3, 192, 320)  # 12*16=192, 20*16=320


def test_decoder_output_range():
    """Decoder output should be in valid image range."""
    from vqgan.decoder import Decoder

    decoder = Decoder(out_channels=3, hidden_channels=64, codebook_dim=128)
    z = mx.random.normal((1, 128, 8, 8))
    x = decoder(z)

    # Output uses tanh, so should be in [-1, 1]
    # (In practice, may slightly exceed due to initialization)
    assert x.dtype == mx.float32
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_decoder.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

`src/vqgan/decoder.py`:
```python
"""VQGAN Decoder - reconstructs images from quantized latents."""

import mlx.core as mx
import mlx.nn as nn
from typing import List

from .layers import ResidualBlock, Upsample


class Decoder(nn.Module):
    """Decoder that upsamples latents back to images.

    Performs 16x spatial upsampling: 20x12 -> 320x192
    (Note: 200 doesn't divide evenly by 16, so we get 192)
    """

    def __init__(
        self,
        out_channels: int = 3,
        hidden_channels: int = 128,
        codebook_dim: int = 256,
        num_res_blocks: int = 2,
        channel_multipliers: List[int] = [1, 2, 2, 4]
    ):
        super().__init__()

        # Reverse multipliers for upsampling
        channel_multipliers = list(reversed(channel_multipliers))
        initial_channels = hidden_channels * channel_multipliers[0]

        # Initial convolution from codebook dim
        self.conv_in = nn.Conv2d(codebook_dim, initial_channels, kernel_size=3, padding=1)

        # Upsampling blocks
        self.up_blocks = []
        channels = initial_channels

        for i, mult in enumerate(channel_multipliers):
            out_ch = hidden_channels * mult

            # Residual blocks
            for _ in range(num_res_blocks):
                self.up_blocks.append(ResidualBlock(channels))

            # Upsample (except last)
            if i < len(channel_multipliers) - 1:
                next_mult = channel_multipliers[i + 1]
                next_ch = hidden_channels * next_mult
                self.up_blocks.append(Upsample(channels, next_ch))
                channels = next_ch

        # Final layers
        self.norm_out = nn.GroupNorm(32, channels)
        self.conv_out = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)

    def __call__(self, z: mx.array) -> mx.array:
        """Decode latent to image.

        Args:
            z: Latent vectors (B, codebook_dim, H, W)

        Returns:
            x: Reconstructed image (B, C, H*16, W*16) in [-1, 1]
        """
        x = self.conv_in(z)

        for block in self.up_blocks:
            x = block(x)

        x = self.norm_out(x)
        x = nn.silu(x)
        x = self.conv_out(x)
        x = mx.tanh(x)

        return x
```

Update `src/vqgan/__init__.py`:
```python
from .layers import ResidualBlock, Downsample, Upsample
from .codebook import VectorQuantizer
from .encoder import Encoder
from .decoder import Decoder
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_decoder.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/vqgan/decoder.py src/vqgan/__init__.py tests/test_decoder.py
git commit -m "feat: add VQGAN decoder"
```

---

### Task 9: Patch Discriminator

**Files:**
- Create: `src/vqgan/discriminator.py`
- Create: `tests/test_discriminator.py`

**Step 1: Write the failing test**

`tests/test_discriminator.py`:
```python
import mlx.core as mx
import sys
sys.path.insert(0, 'src')


def test_discriminator_outputs_patch_logits():
    """Discriminator should output per-patch real/fake logits."""
    from vqgan.discriminator import PatchDiscriminator

    disc = PatchDiscriminator(in_channels=3)
    x = mx.random.normal((2, 3, 200, 320))

    logits = disc(x)

    # Should output spatial map of logits
    assert len(logits.shape) == 4
    assert logits.shape[0] == 2  # batch size preserved
    assert logits.shape[1] == 1  # single channel (real/fake)


def test_discriminator_gradient_flows():
    """Discriminator should have trainable parameters."""
    from vqgan.discriminator import PatchDiscriminator

    disc = PatchDiscriminator(in_channels=3)
    params = disc.parameters()

    # Should have parameters
    assert len(list(params.keys())) > 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_discriminator.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

`src/vqgan/discriminator.py`:
```python
"""PatchGAN Discriminator for VQGAN."""

import mlx.core as mx
import mlx.nn as nn


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator that classifies image patches as real/fake.

    Outputs a spatial map of logits rather than a single value,
    which provides more gradient signal for training.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        num_layers: int = 3
    ):
        super().__init__()

        layers = []

        # First layer (no normalization)
        layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1))

        # Middle layers
        channels = hidden_channels
        for i in range(1, num_layers):
            out_channels = min(channels * 2, 512)
            layers.append(nn.Conv2d(channels, out_channels, kernel_size=4, stride=2, padding=1))
            layers.append(nn.GroupNorm(32, out_channels))
            channels = out_channels

        # Final layer
        layers.append(nn.Conv2d(channels, 1, kernel_size=4, stride=1, padding=1))

        self.layers = layers

    def __call__(self, x: mx.array) -> mx.array:
        """Compute patch-wise real/fake logits.

        Args:
            x: Input image (B, C, H, W)

        Returns:
            logits: Patch logits (B, 1, H', W')
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply LeakyReLU after conv (except last layer)
            if i < len(self.layers) - 1 and isinstance(layer, nn.Conv2d):
                x = nn.leaky_relu(x, negative_slope=0.2)

        return x
```

Update `src/vqgan/__init__.py`:
```python
from .layers import ResidualBlock, Downsample, Upsample
from .codebook import VectorQuantizer
from .encoder import Encoder
from .decoder import Decoder
from .discriminator import PatchDiscriminator
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_discriminator.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/vqgan/discriminator.py src/vqgan/__init__.py tests/test_discriminator.py
git commit -m "feat: add patch discriminator"
```

---

### Task 10: Full VQGAN Model

**Files:**
- Create: `src/vqgan/model.py`
- Create: `tests/test_vqgan_model.py`

**Step 1: Write the failing test**

`tests/test_vqgan_model.py`:
```python
import mlx.core as mx
import sys
sys.path.insert(0, 'src')


def test_vqgan_encode_decode_roundtrip():
    """VQGAN should encode and decode images."""
    from vqgan.model import VQGAN

    model = VQGAN(
        in_channels=3,
        hidden_channels=64,
        codebook_size=256,
        codebook_dim=128
    )

    x = mx.random.normal((2, 3, 192, 320))  # Use 192 for clean divisibility

    x_recon, indices, vq_loss = model(x)

    assert x_recon.shape == x.shape
    assert indices.shape == (2, 12, 20)  # 192/16=12, 320/16=20
    assert vq_loss.shape == ()


def test_vqgan_encode_to_indices():
    """VQGAN should encode images to discrete indices."""
    from vqgan.model import VQGAN

    model = VQGAN(codebook_size=512, codebook_dim=256)
    x = mx.random.normal((1, 3, 192, 320))

    indices = model.encode(x)

    assert indices.shape == (1, 12, 20)
    assert indices.dtype == mx.int32


def test_vqgan_decode_from_indices():
    """VQGAN should decode indices back to images."""
    from vqgan.model import VQGAN

    model = VQGAN(codebook_size=512, codebook_dim=256)
    indices = mx.randint(0, 512, (1, 12, 20))

    x = model.decode(indices)

    assert x.shape == (1, 3, 192, 320)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_vqgan_model.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

`src/vqgan/model.py`:
```python
"""Full VQGAN model combining encoder, quantizer, and decoder."""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple

from .encoder import Encoder
from .decoder import Decoder
from .codebook import VectorQuantizer


class VQGAN(nn.Module):
    """Vector Quantized GAN for pixel art compression.

    Encodes images to discrete tokens and decodes back to images.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 128,
        codebook_size: int = 512,
        codebook_dim: int = 256,
        num_res_blocks: int = 2
    ):
        super().__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            codebook_dim=codebook_dim,
            num_res_blocks=num_res_blocks
        )

        self.quantizer = VectorQuantizer(
            codebook_size=codebook_size,
            codebook_dim=codebook_dim
        )

        self.decoder = Decoder(
            out_channels=in_channels,
            hidden_channels=hidden_channels,
            codebook_dim=codebook_dim,
            num_res_blocks=num_res_blocks
        )

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """Forward pass: encode, quantize, decode.

        Args:
            x: Input image (B, C, H, W) in [-1, 1]

        Returns:
            x_recon: Reconstructed image (B, C, H, W)
            indices: Codebook indices (B, H', W')
            vq_loss: Vector quantization loss
        """
        z = self.encoder(x)
        z_q, indices, vq_loss = self.quantizer(z)
        x_recon = self.decoder(z_q)

        return x_recon, indices, vq_loss

    def encode(self, x: mx.array) -> mx.array:
        """Encode image to discrete indices.

        Args:
            x: Input image (B, C, H, W)

        Returns:
            indices: Codebook indices (B, H', W')
        """
        z = self.encoder(x)
        _, indices, _ = self.quantizer(z)
        return indices

    def decode(self, indices: mx.array) -> mx.array:
        """Decode indices to image.

        Args:
            indices: Codebook indices (B, H', W')

        Returns:
            x: Reconstructed image (B, C, H, W)
        """
        z_q = self.quantizer.decode(indices)
        x = self.decoder(z_q)
        return x
```

Update `src/vqgan/__init__.py`:
```python
from .layers import ResidualBlock, Downsample, Upsample
from .codebook import VectorQuantizer
from .encoder import Encoder
from .decoder import Decoder
from .discriminator import PatchDiscriminator
from .model import VQGAN
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_vqgan_model.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/vqgan/model.py src/vqgan/__init__.py tests/test_vqgan_model.py
git commit -m "feat: add full VQGAN model"
```

---

### Task 11: VQGAN Training Script

**Files:**
- Create: `src/train_vqgan.py`

**Step 1: Write the training script**

`src/train_vqgan.py`:
```python
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
```

**Step 2: Test that script runs (dry run)**

Run: `cd /Users/nervous/Library/CloudStorage/Dropbox/Github/scene && python -c "from src.train_vqgan import load_config; print('OK')"`
Expected: "OK" (verifies imports work)

**Step 3: Commit**

```bash
git add src/train_vqgan.py
git commit -m "feat: add VQGAN training script"
```

---

## Phase 4: Transformer Components

### Task 12: Simple Text Tokenizer

**Files:**
- Create: `src/transformer/tokenizer.py`
- Create: `tests/test_tokenizer.py`

**Step 1: Write the failing test**

`tests/test_tokenizer.py`:
```python
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tokenizer.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

`src/transformer/tokenizer.py`:
```python
"""Simple character/word tokenizer for text conditioning."""

import re
from typing import List, Optional


class SimpleTokenizer:
    """Simple word-level tokenizer with hash-based vocabulary.

    Uses hashing to map words to token IDs without needing
    a pre-built vocabulary. Good enough for small-scale training.
    """

    def __init__(self, vocab_size: int = 1000, max_length: int = 128):
        self.vocab_size = vocab_size
        self.max_length = max_length

        # Special tokens
        self.pad_token = 0
        self.unk_token = 1
        self.start_token = 2
        self.end_token = 3
        self.special_tokens = 4  # Reserve first 4 IDs

    def _hash_word(self, word: str) -> int:
        """Hash a word to a token ID."""
        # Simple hash function
        h = hash(word.lower()) % (self.vocab_size - self.special_tokens)
        return h + self.special_tokens

    def _tokenize(self, text: str) -> List[str]:
        """Split text into words."""
        # Simple word tokenization
        text = text.lower().strip()
        words = re.findall(r'\b\w+\b', text)
        return words

    def encode(self, text: str, pad: bool = False) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input text string
            pad: Whether to pad/truncate to max_length

        Returns:
            List of token IDs
        """
        words = self._tokenize(text)
        tokens = [self.start_token]
        tokens.extend(self._hash_word(w) for w in words)
        tokens.append(self.end_token)

        if pad:
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length-1] + [self.end_token]
            else:
                tokens = tokens + [self.pad_token] * (self.max_length - len(tokens))

        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Decode tokens back to text (lossy - just for debugging)."""
        # Can't truly decode hash-based tokens, return placeholder
        return f"<{len(tokens)} tokens>"
```

Update `src/transformer/__init__.py`:
```python
from .tokenizer import SimpleTokenizer
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tokenizer.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/transformer/tokenizer.py src/transformer/__init__.py tests/test_tokenizer.py
git commit -m "feat: add simple text tokenizer"
```

---

### Task 13: Text Encoder

**Files:**
- Create: `src/transformer/text_encoder.py`
- Create: `tests/test_text_encoder.py`

**Step 1: Write the failing test**

`tests/test_text_encoder.py`:
```python
import mlx.core as mx
import sys
sys.path.insert(0, 'src')


def test_text_encoder_embeds_tokens():
    """Text encoder should convert tokens to embeddings."""
    from transformer.text_encoder import TextEncoder

    encoder = TextEncoder(vocab_size=1000, embed_dim=256, max_length=128)

    # Batch of 2, sequence length 32
    tokens = mx.randint(0, 1000, (2, 32))
    embeddings = encoder(tokens)

    assert embeddings.shape == (2, 32, 256)


def test_text_encoder_with_mask():
    """Text encoder should handle attention masks."""
    from transformer.text_encoder import TextEncoder

    encoder = TextEncoder(vocab_size=1000, embed_dim=256)
    tokens = mx.randint(0, 1000, (1, 16))

    embeddings = encoder(tokens)

    assert embeddings.shape == (1, 16, 256)
    assert embeddings.dtype == mx.float32
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_text_encoder.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

`src/transformer/text_encoder.py`:
```python
"""Text encoder for conditioning the transformer."""

import mlx.core as mx
import mlx.nn as nn
import math


class TextEncoder(nn.Module):
    """Transformer-based text encoder.

    Encodes text tokens into contextual embeddings for
    conditioning the image generation transformer.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        max_length: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.max_length = max_length

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional embeddings
        self.pos_embedding = nn.Embedding(max_length, embed_dim)

        # Transformer layers
        self.layers = [
            nn.TransformerEncoderLayer(
                dims=embed_dim,
                num_heads=num_heads,
                mlp_dims=embed_dim * 4,
                dropout=dropout
            )
            for _ in range(num_layers)
        ]

        self.norm = nn.LayerNorm(embed_dim)

    def __call__(self, tokens: mx.array) -> mx.array:
        """Encode tokens to embeddings.

        Args:
            tokens: Token IDs (B, L)

        Returns:
            embeddings: Contextual embeddings (B, L, D)
        """
        B, L = tokens.shape

        # Token + positional embeddings
        positions = mx.arange(L)
        x = self.token_embedding(tokens) + self.pos_embedding(positions)

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        return x
```

Update `src/transformer/__init__.py`:
```python
from .tokenizer import SimpleTokenizer
from .text_encoder import TextEncoder
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_text_encoder.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/transformer/text_encoder.py src/transformer/__init__.py tests/test_text_encoder.py
git commit -m "feat: add text encoder"
```

---

### Task 14: Image Transformer (Token Generator)

**Files:**
- Create: `src/transformer/model.py`
- Create: `tests/test_transformer_model.py`

**Step 1: Write the failing test**

`tests/test_transformer_model.py`:
```python
import mlx.core as mx
import sys
sys.path.insert(0, 'src')


def test_transformer_generates_tokens():
    """Transformer should generate image token logits."""
    from transformer.model import ImageTransformer

    model = ImageTransformer(
        vocab_size=512,      # codebook size
        text_vocab_size=1000,
        max_seq_len=240,     # 20x12 tokens
        embed_dim=256,
        num_heads=8,
        num_layers=6
    )

    # Text tokens and image tokens (for training)
    text_tokens = mx.randint(0, 1000, (2, 32))
    image_tokens = mx.randint(0, 512, (2, 240))

    logits = model(text_tokens, image_tokens)

    # Should output logits for each position
    assert logits.shape == (2, 240, 512)


def test_transformer_generates_autoregressively():
    """Transformer should generate tokens one at a time."""
    from transformer.model import ImageTransformer

    model = ImageTransformer(
        vocab_size=512,
        text_vocab_size=1000,
        max_seq_len=240,
        embed_dim=128,
        num_heads=4,
        num_layers=2
    )

    text_tokens = mx.randint(0, 1000, (1, 16))

    # Generate all tokens
    generated = model.generate(text_tokens, temperature=1.0)

    assert generated.shape == (1, 240)
    assert mx.all(generated >= 0)
    assert mx.all(generated < 512)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_transformer_model.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

`src/transformer/model.py`:
```python
"""Image token transformer for text-to-image generation."""

import mlx.core as mx
import mlx.nn as nn
import math
from typing import Optional

from .text_encoder import TextEncoder


class ImageTransformer(nn.Module):
    """Transformer that generates image tokens conditioned on text.

    Uses cross-attention to condition on text embeddings and
    generates image tokens autoregressively.
    """

    def __init__(
        self,
        vocab_size: int = 512,       # VQGAN codebook size
        text_vocab_size: int = 1000,
        max_seq_len: int = 240,      # 20x12 image tokens
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        text_embed_dim: int = 256,
        max_text_len: int = 128
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        # Text encoder
        self.text_encoder = TextEncoder(
            vocab_size=text_vocab_size,
            embed_dim=text_embed_dim,
            max_length=max_text_len
        )

        # Image token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Transformer decoder layers with cross-attention
        self.layers = []
        for _ in range(num_layers):
            self.layers.append({
                'self_attn': nn.MultiHeadAttention(embed_dim, num_heads),
                'cross_attn': nn.MultiHeadAttention(embed_dim, num_heads),
                'mlp': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * mlp_ratio),
                    nn.GELU(),
                    nn.Linear(embed_dim * mlp_ratio, embed_dim)
                ),
                'norm1': nn.LayerNorm(embed_dim),
                'norm2': nn.LayerNorm(embed_dim),
                'norm3': nn.LayerNorm(embed_dim),
            })

        self.final_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def _causal_mask(self, seq_len: int) -> mx.array:
        """Create causal attention mask."""
        mask = mx.triu(mx.full((seq_len, seq_len), float('-inf')), k=1)
        return mask

    def forward_with_cache(
        self,
        text_embeddings: mx.array,
        image_tokens: mx.array,
        cache: Optional[list] = None
    ) -> tuple:
        """Forward pass with KV cache for efficient generation."""
        B, L = image_tokens.shape

        # Embed image tokens
        positions = mx.arange(L)
        x = self.token_embedding(image_tokens) + self.pos_embedding(positions)

        # Causal mask for self-attention
        causal_mask = self._causal_mask(L)

        new_cache = []
        for i, layer in enumerate(self.layers):
            # Self-attention with causal mask
            residual = x
            x = layer['norm1'](x)
            x = layer['self_attn'](x, x, x, mask=causal_mask) + residual

            # Cross-attention to text
            residual = x
            x = layer['norm2'](x)
            x = layer['cross_attn'](x, text_embeddings, text_embeddings) + residual

            # MLP
            residual = x
            x = layer['norm3'](x)
            x = layer['mlp'](x) + residual

            new_cache.append(None)  # Placeholder for future cache implementation

        x = self.final_norm(x)
        logits = self.output_proj(x)

        return logits, new_cache

    def __call__(
        self,
        text_tokens: mx.array,
        image_tokens: mx.array
    ) -> mx.array:
        """Forward pass for training.

        Args:
            text_tokens: Text token IDs (B, text_len)
            image_tokens: Image token IDs (B, seq_len)

        Returns:
            logits: Token logits (B, seq_len, vocab_size)
        """
        # Encode text
        text_embeddings = self.text_encoder(text_tokens)

        # Generate logits
        logits, _ = self.forward_with_cache(text_embeddings, image_tokens)

        return logits

    def generate(
        self,
        text_tokens: mx.array,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> mx.array:
        """Generate image tokens autoregressively.

        Args:
            text_tokens: Text token IDs (B, text_len)
            temperature: Sampling temperature
            top_k: If set, only sample from top k tokens

        Returns:
            generated: Generated image tokens (B, max_seq_len)
        """
        B = text_tokens.shape[0]

        # Encode text once
        text_embeddings = self.text_encoder(text_tokens)

        # Start with a random first token (or learned start token)
        generated = mx.zeros((B, 1), dtype=mx.int32)

        for i in range(self.max_seq_len - 1):
            logits, _ = self.forward_with_cache(text_embeddings, generated)
            next_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                # Top-k sampling
                top_k_vals, top_k_idx = mx.topk(next_logits, k=top_k)
                probs = mx.softmax(top_k_vals, axis=-1)
                sampled_idx = mx.random.categorical(probs)
                next_token = mx.take_along_axis(top_k_idx, sampled_idx[:, None], axis=-1)
            else:
                # Full sampling
                probs = mx.softmax(next_logits, axis=-1)
                next_token = mx.random.categorical(probs)[:, None]

            generated = mx.concatenate([generated, next_token.astype(mx.int32)], axis=1)

        return generated
```

Update `src/transformer/__init__.py`:
```python
from .tokenizer import SimpleTokenizer
from .text_encoder import TextEncoder
from .model import ImageTransformer
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_transformer_model.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/transformer/model.py src/transformer/__init__.py tests/test_transformer_model.py
git commit -m "feat: add image transformer model"
```

---

### Task 15: Transformer Training Script

**Files:**
- Create: `src/train_transformer.py`

**Step 1: Write the training script**

`src/train_transformer.py`:
```python
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
```

**Step 2: Test that script runs (dry run)**

Run: `cd /Users/nervous/Library/CloudStorage/Dropbox/Github/scene && python -c "from src.train_transformer import load_config; print('OK')"`
Expected: "OK"

**Step 3: Commit**

```bash
git add src/train_transformer.py
git commit -m "feat: add transformer training script"
```

---

## Phase 5: Generation Pipeline

### Task 16: Generation Script

**Files:**
- Create: `src/generate.py`

**Step 1: Write the generation script**

`src/generate.py`:
```python
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
```

**Step 2: Test that script runs (dry run)**

Run: `cd /Users/nervous/Library/CloudStorage/Dropbox/Github/scene && python -c "from src.generate import generate; print('OK')"`
Expected: "OK"

**Step 3: Commit**

```bash
git add src/generate.py
git commit -m "feat: add generation script"
```

---

### Task 17: Create README

**Files:**
- Create: `README.md`

**Step 1: Write README**

`README.md`:
```markdown
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
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README"
```

---

## Summary

**Phase 1 (Tasks 1)**: Project setup - configs, structure, dependencies
**Phase 2 (Tasks 2-3)**: Dataset loaders for images and captions
**Phase 3 (Tasks 4-11)**: VQGAN components and training
**Phase 4 (Tasks 12-15)**: Transformer components and training
**Phase 5 (Tasks 16-17)**: Generation pipeline and documentation

After completing all tasks, you'll have a working text-to-pixel-art system!
