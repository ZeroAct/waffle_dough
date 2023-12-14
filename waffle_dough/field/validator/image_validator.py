from typing import Union

from waffle_dough.exception import FieldValidationError
from waffle_dough.type import SplitType


def validate_width(v: int) -> int:
    if v <= 0:
        raise FieldValidationError(f"width must be positive integer: {v}")
    return v


def validate_height(v: int) -> int:
    if v <= 0:
        raise FieldValidationError(f"height must be positive integer: {v}")
    return v


def validate_split(v: Union[str, SplitType]) -> str:
    if v not in list(SplitType):
        raise FieldValidationError(f"split must be one of {list(SplitType)}: {v}")
    return v.lower()
