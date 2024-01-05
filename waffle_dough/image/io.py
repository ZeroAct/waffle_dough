from pathlib import Path
from typing import Union

import cv2
import numpy as np

from waffle_dough.type import ColorType

COLOR_TYPE_MAP = {
    ColorType.GRAY: "GRAY",
    ColorType.RGB: "RGB",
    ColorType.BGR: "BGR",
}


def cv2_cvt_color(
    image: np.ndarray, src_type: Union[str, ColorType], dst_type: Union[str, ColorType]
) -> np.ndarray:
    if src_type == dst_type:
        return image

    src_type = COLOR_TYPE_MAP.get(src_type)
    dst_type = COLOR_TYPE_MAP.get(dst_type)

    return cv2.cvtColor(image, getattr(cv2, f"COLOR_{src_type}2{dst_type}"))


def cv2_imread(
    path: Union[str, Path], color_type: Union[str, ColorType] = ColorType.BGR
) -> np.ndarray:
    image = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    image = cv2_cvt_color(image, ColorType.BGR, color_type)
    return image


def cv2_imwrite(
    image: np.ndarray,
    path: Union[str, Path],
    create_directory: bool = False,
    color_type: Union[str, ColorType] = ColorType.BGR,
):
    output_path = Path(path)
    if create_directory:
        output_path.make_directory()

    save_type = output_path.suffix
    bgr_image = cv2_cvt_color(image, color_type, ColorType.BGR)
    ret, img_arr = cv2.imencode(save_type, bgr_image)
    if ret:
        with open(str(output_path), mode="w+b") as f:
            img_arr.tofile(f)
    else:
        raise ValueError(f"Failed to save image: {path}")
