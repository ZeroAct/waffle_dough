from typing import Union

from waffle_dough.type import SplitType


def validate_width(v: int) -> int:
    if v <= 0:
        raise ValueError(f"width must be positive integer: {v}")
    return v


def validate_height(v: int) -> int:
    if v <= 0:
        raise ValueError(f"height must be positive integer: {v}")
    return v


def validate_split(v: Union[str, SplitType]) -> str:
    if v not in list(SplitType):
        raise ValueError(f"split must be one of {list(SplitType)}: {v}")
    return v.lower()
