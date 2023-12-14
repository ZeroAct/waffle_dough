from waffle_utils.enum import StrEnum as BaseType


class BoxType(BaseType):
    """Box Type Class

    BoxType.XYXY -> x1, y1, x2, y2
    BoxType.XYWH -> x1, y1, w, h
    BoxType.CXCYWH -> cx, cy, w, h
    """

    XYXY = "xyxy"
    XYWH = "xywh"
    CXCYWH = "cxcywh"


class SegmentationType(BaseType):
    """Segmentation Type Class

    SegmentationType.POLYGON -> polygon
    SegmentationType.MASK -> mask
    SegmentationType.RLE -> rle
    """

    POLYGON = "polygon"
    MASK = "mask"
    RLE = "rle"
