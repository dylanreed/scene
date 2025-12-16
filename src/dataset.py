"""Dataset utilities for loading pixel art images."""

import os
from pathlib import Path
from typing import Tuple, List, Iterator
import numpy as np
from PIL import Image
import mlx.core as mx
import json


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
