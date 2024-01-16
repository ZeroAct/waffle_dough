from typing import Union

import numpy as np

from waffle_dough.exception import FieldValidationError


def validate_bbox(v: list[float]) -> list[float]:
    if v:
        if len(v) != 4:
            raise FieldValidationError("the length of bbox should be 4.")
        if not all([isinstance(v_, (float)) for v_ in v]):
            raise FieldValidationError("the bbox values should be float.")
        if not all([0 <= v_ < 1 for v_ in v[:2]]):
            raise FieldValidationError("the [left, top] should be in [0, 1).")
        if not all([0 < v_ <= 1 for v_ in v[2:]]):
            raise FieldValidationError("the [width, height] should be in (0, 1].")
    return v


def validate_segmentation(
    v: Union[dict, list[list[float]], np.ndarray],
) -> Union[list[list[float]], np.ndarray]:
    if v:
        if not isinstance(v, (list, np.ndarray)):
            raise FieldValidationError("segmentation should be list or np.ndarray.")

        for segment in v:
            if len(segment) % 2 != 0:
                raise FieldValidationError("the length of segmentation should be divisible by 2.")
            if len(segment) < 6:
                raise FieldValidationError(
                    "the length of segmentation should be greater than or equal to 6."
                )
    return v


def validate_area(v: float) -> float:
    if v:
        if v < 0:
            raise FieldValidationError("area should be greater than or equal to 0.")
    return v


def validate_keypoints(v: list[float]) -> list[float]:
    if v:
        if len(v) % 3 != 0:
            raise FieldValidationError("the length of keypoints should be divisible by 3.")
    return v


def validate_num_keypoints(v: int) -> int:
    if v:
        if v < 0:
            raise FieldValidationError("num_keypoints should be greater than or equal to 0.")
    return v


def validate_caption(v: str) -> str:
    if v:
        if not isinstance(v, str):
            raise FieldValidationError("caption should be str.")
    return v


def validate_value(v: Union[int, float]) -> float:
    if v:
        if not isinstance(v, (int, float)):
            raise FieldValidationError("value should be int or float.")
        v = float(v)
    return v


def validate_iscrowd(v: int) -> int:
    if v:
        if v not in [0, 1]:
            raise FieldValidationError("iscrowd should be 0 or 1.")
    return v


def validate_score(v: float) -> float:
    if v:
        if v < 0 or v > 1:
            raise FieldValidationError("score should be in [0, 1].")
    return v


def validate_is_prediction(v: bool) -> bool:
    if v:
        if not isinstance(v, bool):
            raise FieldValidationError("is_prediction should be bool.")
    return v
