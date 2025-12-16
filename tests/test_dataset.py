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
