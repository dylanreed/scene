# Training Metrics Design

## Overview

Add FID scoring and loss history logging to track VQGAN training quality over time.

## Components

### MetricsLogger

Writes training metrics to CSV after each epoch.

**File:** `src/metrics.py`

**Interface:**
```python
class MetricsLogger:
    def __init__(self, output_path: str)
    def log_epoch(self, epoch: int, losses: dict, duration: float, fid: float = None)
```

**CSV columns:** epoch, g_loss, d_loss, recon_loss, vq_loss, fid, duration_sec, samples_per_sec

### FIDScorer

Computes Fréchet Inception Distance between generated and real images.

**File:** `src/metrics.py`

**Interface:**
```python
class FIDScorer:
    def __init__(self, num_real_samples: int = 500)
    def cache_real_features(self, dataloader)
    def compute_fid(self, model, dataloader, num_samples: int = 500) -> float
```

**Behavior:**
- Uses InceptionV3 to extract features
- Caches real image features at training start
- Generates samples and computes Fréchet distance
- Lower FID = better quality

### Config Options

```yaml
metrics:
  log_file: "outputs/training_metrics.csv"
  fid_every: 5          # compute FID every N epochs
  fid_samples: 500      # samples for FID computation
  fid_full_at_end: true # full dataset FID after training
```

## Integration

Changes to `train_vqgan.py`:

1. Create `MetricsLogger` and `FIDScorer` at training start
2. Cache real features once before training loop
3. Track epoch duration with timing
4. Compute FID every N epochs
5. Log all metrics after each epoch
6. Compute full-dataset FID after training completes

## Output

`outputs/training_metrics.csv` - can be plotted with matplotlib, pandas, or spreadsheet software.
