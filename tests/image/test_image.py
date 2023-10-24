import cv2
import numpy as np
import pytest

from waffle_dough.image import Image
from waffle_dough.type import ColorType


@pytest.fixture
def rgb_image_path(tmpdir):
    image = np.random.randint(0, 255, size=(100, 200, 3), dtype=np.uint8)
    image_path = tmpdir.join("test_image.jpg")

    cv2.imwrite(str(image_path), image)

    return image_path


@pytest.fixture
def gray_image_path(tmpdir):
    image = np.random.randint(0, 255, size=(100, 200), dtype=np.uint8)
    image_path = tmpdir.join("test_image.jpg")

    cv2.imwrite(str(image_path), image)

    return image_path


@pytest.mark.parametrize(
    "image_path",
    [
        "rgb_image_path",
        "gray_image_path",
    ],
)
def test_image(image_path, tmpdir, request):
    image_path = request.getfixturevalue(image_path)

    cv_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    image = Image.load(str(image_path))

    # test io
    assert np.array_equal(cv_image, image)

    tmp_image_path = tmpdir.join("tmp_image.jpg")
    image.save(tmp_image_path)

    # TODO: cannot generalize this assertion
    # assert np.mean(image - Image.load(tmp_image_path)) / 255 < 0.01

    # test attributes
    assert image.color_type == ColorType.RGB
    assert image.shape == (100, 200, 3)
    assert image.height == 100
    assert image.width == 200
    assert image.channels == 3
    assert image.resolution == (200, 100)
    assert image.aspect_ratio == 2.0
