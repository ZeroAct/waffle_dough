from typing import Union

from waffle_dough.type.annotation_type import BoxType


def x1y1x2y2_to_x1y1wh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def x1y1x2y2_to_cxcywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


def x1y1wh_to_x1y1x2y2(box: list[float]) -> list[float]:
    x1, y1, w, h = box
    return [x1, y1, x1 + w, y1 + h]


def x1y1wh_to_cxcywh(box: list[float]) -> list[float]:
    x1, y1, w, h = box
    return [x1 + w / 2, y1 + h / 2, w, h]


def cxcywh_to_x1y1x2y2(box: list[float]) -> list[float]:
    cx, cy, w, h = box
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def cxcywh_to_x1y1wh(box: list[float]) -> list[float]:
    cx, cy, w, h = box
    return [cx - w / 2, cy - h / 2, w, h]


def convert_box(
    box: list[float], src_type: Union[str, BoxType], dst_type: Union[str, BoxType]
) -> list[float]:
    """Convert box from src_type to dst_type.

    Args:
        box (list[float]): box
        src_type (Union[str, BoxType]): source type
        dst_type (Union[str, BoxType]): destination type

    Returns:
        list[float]: converted bbox
    """

    if src_type == dst_type:
        return box

    if src_type == BoxType.XYXY:
        if dst_type == BoxType.XYWH:
            return x1y1x2y2_to_x1y1wh(box)
        elif dst_type == BoxType.CXCYWH:
            return x1y1x2y2_to_cxcywh(box)
    elif src_type == BoxType.XYWH:
        if dst_type == BoxType.XYXY:
            return x1y1wh_to_x1y1x2y2(box)
        elif dst_type == BoxType.CXCYWH:
            return x1y1wh_to_cxcywh(box)
    elif src_type == BoxType.CXCYWH:
        if dst_type == BoxType.XYXY:
            return cxcywh_to_x1y1x2y2(box)
        elif dst_type == BoxType.XYWH:
            return cxcywh_to_x1y1wh(box)
    else:
        raise ValueError(f"Cannot convert bbox from {src_type} to {dst_type}")


def get_box_area(box: list[float], box_type: Union[str, BoxType]) -> float:
    """Get area from box.

    Args:
        box (list[float]): box
        box_type (Union[str, BoxType]): box type

    Returns:
        float: area
    """

    _, _, w, h = convert_box(box, box_type, BoxType.XYWH)
    return w * h
