"""Training metrics: loss logging and FID scoring."""

import csv
import os
import time
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np

# Optional torch import for FID
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Optional scipy import for FID calculation
try:
    from scipy import linalg
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class MetricsLogger:
    """Log training metrics to CSV file."""

    COLUMNS = [
        'epoch', 'g_loss', 'd_loss', 'recon_loss', 'vq_loss',
        'fid', 'duration_sec', 'samples_per_sec'
    ]

    def __init__(self, output_path: str):
        """Initialize logger and create CSV with headers.

        Args:
            output_path: Path to CSV file
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write headers if file doesn't exist
        if not self.output_path.exists():
            with open(self.output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.COLUMNS)

    def log_epoch(
        self,
        epoch: int,
        losses: Dict[str, float],
        duration: float,
        num_samples: int,
        fid: Optional[float] = None
    ):
        """Log metrics for one epoch.

        Args:
            epoch: Epoch number
            losses: Dict with g_loss, d_loss, recon_loss, vq_loss
            duration: Epoch duration in seconds
            num_samples: Number of samples processed
            fid: FID score (None if not computed)
        """
        samples_per_sec = num_samples / duration if duration > 0 else 0

        row = [
            epoch,
            f"{losses.get('g_loss', 0):.6f}",
            f"{losses.get('d_loss', 0):.6f}",
            f"{losses.get('recon_loss', 0):.6f}",
            f"{losses.get('vq_loss', 0):.6f}",
            f"{fid:.4f}" if fid is not None else "",
            f"{duration:.2f}",
            f"{samples_per_sec:.2f}"
        ]

        with open(self.output_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)


class FIDScorer:
    """Compute Fréchet Inception Distance for image quality assessment.

    FID measures the similarity between generated and real image distributions
    using features from a pretrained InceptionV3 network.

    Lower FID = better quality (0 = identical distributions)

    Requires: torch, torchvision, scipy
    """

    def __init__(self, device: str = None):
        """Initialize FID scorer.

        Args:
            device: torch device ('cuda', 'mps', 'cpu'). Auto-detected if None.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "FID scoring requires PyTorch. Install with: pip install torch torchvision"
            )
        if not SCIPY_AVAILABLE:
            raise RuntimeError(
                "FID scoring requires scipy. Install with: pip install scipy"
            )

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = torch.device(device)
        self.model = self._load_inception()
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Cached real features
        self.real_mu: Optional[np.ndarray] = None
        self.real_sigma: Optional[np.ndarray] = None

    def _load_inception(self):
        """Load InceptionV3 and modify for feature extraction."""
        inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        inception.fc = nn.Identity()  # Remove final classification layer
        inception.eval()
        inception.to(self.device)
        return inception

    def _extract_features(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract InceptionV3 features from images.

        Args:
            images: List of numpy arrays (H, W, 3) in range [0, 255]

        Returns:
            Features array of shape (N, 2048)
        """
        from PIL import Image

        features = []
        batch_size = 32

        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]

                # Convert to PIL and apply transforms
                tensors = []
                for img in batch_images:
                    if img.dtype != np.uint8:
                        img = (img * 255).astype(np.uint8)
                    pil_img = Image.fromarray(img)
                    tensors.append(self.transform(pil_img))

                batch = torch.stack(tensors).to(self.device)

                # Extract features
                feat = self.model(batch)
                features.append(feat.cpu().numpy())

        return np.concatenate(features, axis=0)

    def _compute_statistics(self, features: np.ndarray):
        """Compute mean and covariance of features."""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def _frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Compute Fréchet distance between two Gaussians."""
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

    def cache_real_features(self, images: List[np.ndarray]):
        """Cache statistics from real images (call once at training start).

        Args:
            images: List of numpy arrays (H, W, 3) in range [0, 255]
        """
        print(f"Caching features for {len(images)} real images...")
        features = self._extract_features(images)
        self.real_mu, self.real_sigma = self._compute_statistics(features)
        print("Real image features cached.")

    def compute_fid(self, generated_images: List[np.ndarray]) -> float:
        """Compute FID between generated images and cached real images.

        Args:
            generated_images: List of numpy arrays (H, W, 3) in range [0, 255]

        Returns:
            FID score (lower is better)
        """
        if self.real_mu is None or self.real_sigma is None:
            raise RuntimeError("Must call cache_real_features() first")

        features = self._extract_features(generated_images)
        gen_mu, gen_sigma = self._compute_statistics(features)

        fid = self._frechet_distance(self.real_mu, self.real_sigma, gen_mu, gen_sigma)
        return float(fid)


def is_fid_available() -> bool:
    """Check if FID scoring is available (torch and scipy installed)."""
    return TORCH_AVAILABLE and SCIPY_AVAILABLE
