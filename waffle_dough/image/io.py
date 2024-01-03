from pathlib import Path
from typing import Union

import cv2
import numpy as np

from waffle_dough.type import ColorType, get_color_types


def cv2_imread(
    path: Union[str, Path], color_type: Union[str, ColorType] = ColorType.BGR
) -> np.ndarray:
    image = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    if color_type == ColorType.GRAY:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, axis=-1)
    elif color_type == ColorType.RGB:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_type == ColorType.BGR:
        pass

    return image


def cv2_imwrite(image: np.ndarray, path: Union[str, Path], create_directory: bool = False) -> None:
    output_path = Path(path)
    if create_directory:
        output_path.make_directory()

    save_type = output_path.suffix
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ret, img_arr = cv2.imencode(save_type, bgr_image)
    if ret:
        with open(str(output_path), mode="w+b") as f:
            img_arr.tofile(f)
    else:
        raise ValueError(f"Failed to save image: {path}")
