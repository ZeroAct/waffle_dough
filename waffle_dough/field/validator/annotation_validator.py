from typing import Optional, Union

import numpy as np

from waffle_dough.math.box import convert_box, get_box_area
from waffle_dough.math.segmentation import convert_segmentation
from waffle_dough.type import BoxType, SegmentationType, TaskType


def validate_bbox(v: list[float], box_type: Union[str, BoxType] = BoxType.XYWH) -> list[float]:
    if v:
        if box_type != BoxType.XYWH:
            v = convert_box(v, box_type, BoxType.XYWH)
        if len(v) != 4:
            raise ValueError("the length of bbox should be 4.")
        if v[2] <= 0 or v[3] <= 0:
            raise ValueError("the width and height of bbox should be greater than 0.")
    return v


def validate_segmentation(
    v: Union[list[list[float]], np.ndarray],
) -> Union[list[list[float]], np.ndarray]:
    if v:
        if isinstance(v, dict):
            if "counts" not in v or "size" not in v:
                raise ValueError(f"the segmentation should have 'counts' and 'size'. Given {v}")
            v = convert_segmentation(v, SegmentationType.RLE, SegmentationType.POLYGON)
        elif isinstance(v, np.ndarray):
            v = convert_segmentation(v, SegmentationType.MASK, SegmentationType.POLYGON)

        for segment in v:
            if len(segment) % 2 != 0:
                raise ValueError("the length of segmentation should be divisible by 2.")
            if len(segment) < 6:
                raise ValueError("the length of segmentation should be greater than or equal to 6.")
    return v


def validate_area(v: float) -> float:
    if v:
        if v < 0:
            raise ValueError("area should be greater than or equal to 0.")
    return v


def validate_keypoints(v: list[float]) -> list[float]:
    if v:
        if len(v) % 3 != 0:
            raise ValueError("the length of keypoints should be divisible by 3.")
    return v


def validate_num_keypoints(v: int) -> int:
    if v:
        if v < 0:
            raise ValueError("num_keypoints should be greater than or equal to 0.")
    return v


def validate_caption(v: str) -> str:
    if v:
        if not isinstance(v, str):
            raise ValueError("caption should be str.")
    return v


def validate_value(v: Union[int, float]) -> float:
    if v:
        if not isinstance(v, (int, float)):
            raise ValueError("value should be int or float.")
        v = float(v)
    return v


def validate_iscrowd(v: int) -> int:
    if v:
        if v not in [0, 1]:
            raise ValueError("iscrowd should be 0 or 1.")
    return v


def validate_score(v: float) -> float:
    if v:
        if v < 0 or v > 1:
            raise ValueError("score should be in [0, 1].")
    return v


def validate_is_prediction(v: bool) -> bool:
    if v:
        if not isinstance(v, bool):
            raise ValueError("is_prediction should be bool.")
    return v
