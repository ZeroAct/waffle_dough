from pathlib import Path
from typing import Union

import cv2
import numpy as np
from waffle_utils.file import io

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
    path: Union[str, Path],
    image: np.ndarray,
    create_directory: bool = False,
    color_type: Union[str, ColorType] = ColorType.BGR,
):
    output_path = Path(path)
    if create_directory:
        io.make_directory(output_path.parent)

    save_type = output_path.suffix
    bgr_image = cv2_cvt_color(image, color_type, ColorType.BGR)
    ret, img_arr = cv2.imencode(save_type, bgr_image)
    if ret:
        with open(str(output_path), mode="w+b") as f:
            img_arr.tofile(f)
    else:
        raise ValueError(f"Failed to save image: {path}")


def cv2_imshow(
    window_name: str,
    image: np.ndarray,
    delay: int = 0,
):
    cv2.imshow(window_name, image)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()
