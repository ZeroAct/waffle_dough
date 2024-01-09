from typing import Union

import numpy as np

from waffle_dough.field import AnnotationInfo, CategoryInfo, ImageInfo
from waffle_dough.image.io import cv2_imread
from waffle_dough.type import ColorType


class Data:
    def __init__(
        self,
        image: np.ndarray,
        image_path: str,
        image_info: ImageInfo,
        annotations: list[AnnotationInfo],
        categories: list[CategoryInfo],
    ):
        self.image = image
        self.image_path = image_path
        self.image_info = image_info
        self.annotations = annotations
        self.categories = categories

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Data(image={self.image.shape}, image_path={self.image_path}, image_info={self.image_info}, annotations={self.annotations}, categories={self.categories})"


class Iterator:
    def __init__(
        self,
        mapper: dict[dict],
        preprocess=None,
        augment=None,
        color_type: Union[str, ColorType] = ColorType.RGB,
    ):
        self.mapper = mapper
        self.image_ids = list(mapper.keys())
        self.preprocess = preprocess
        self.augment = augment
        self.color_type = color_type

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx) -> Data:
        image_id = self.image_ids[idx]

        image_path = self.mapper[image_id]["image_path"]
        image = cv2_imread(image_path, color_type=self.color_type)
        if self.preprocess is not None:
            image = self.preprocess(image)
        if self.augment is not None:
            image = self.augment(image)

        image_info: ImageInfo = self.mapper[image_id]["image_info"]
        annotations: list[AnnotationInfo] = self.mapper[image_id]["annotations"]
        categories: list[CategoryInfo] = self.mapper[image_id]["categories"]

        return Data(image, image_path, image_info, annotations, categories)
