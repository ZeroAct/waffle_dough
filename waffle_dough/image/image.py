from pathlib import Path
from typing import Union

import cv2
import numpy as np
from waffle_utils.validator import setter_type_validator

from waffle_dough.type import ColorType, get_color_types


class Image(np.ndarray):
    def __new__(cls, image: np.ndarray, color_type: Union[str, ColorType] = ColorType.RGB):
        image = image.view(cls)
        image.color_type = color_type
        return image

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.color_type = getattr(obj, "color_type", ColorType.RGB)

    @classmethod
    def load(cls, path: str, color_type: Union[str, ColorType] = ColorType.RGB) -> "Image":
        image = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        if color_type == ColorType.GRAY:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.expand_dims(image, axis=-1)
        elif color_type == ColorType.RGB:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif color_type == ColorType.BGR:
            pass

        return cls(image, color_type=color_type)

    def save(self, path: str, create_directory: bool = False) -> None:
        output_path = Path(path)
        if create_directory:
            output_path.make_directory()

        save_type = output_path.suffix
        bgr_image = cv2.cvtColor(self, cv2.COLOR_RGB2BGR)
        ret, img_arr = cv2.imencode(save_type, bgr_image)
        if ret:
            with open(str(output_path), mode="w+b") as f:
                img_arr.tofile(f)
        else:
            raise ValueError(f"Failed to save image: {path}")

    @property
    def color_type(self) -> str:
        return self._color_type

    @color_type.setter
    @setter_type_validator(ColorType, strict=False)
    def color_type(self, color_type: Union[str, ColorType]) -> None:
        if color_type not in list(ColorType):
            raise ValueError(f"Invalid color type: {color_type}. choose from {get_color_types}")
        self._color_type = color_type.value

    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def channels(self) -> int:
        return self.shape[2]

    @property
    def resolution(self) -> tuple:
        return (self.width, self.height)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height
