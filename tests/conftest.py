import cv2
import numpy as np
import pytest


@pytest.fixture
def sample_image_paths(tmpdir) -> list[str]:
    images = []
    for i in range(10):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        image_path = tmpdir.join(f"image_{i}.jpg")
        cv2.imwrite(str(image_path), image)
        images.append(str(image_path))
    return images
