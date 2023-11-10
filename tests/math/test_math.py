import numpy as np
import pytest

from waffle_dough.math.box import convert_box, get_box_area
from waffle_dough.math.segmentation import (
    convert_segmentation,
    get_segmentation_area,
    get_segmentation_box,
)
from waffle_dough.type.annotation_type import BoxType, SegmentationType


@pytest.mark.parametrize(
    "box, src_type, dst_type, converted_box, area",
    [
        ([0, 0, 100, 100], BoxType.XYXY, BoxType.XYXY, [0, 0, 100, 100], 10000),
        ([0, 0, 100, 100], BoxType.XYXY, BoxType.XYWH, [0, 0, 100, 100], 10000),
        ([0, 0, 100, 100], BoxType.XYXY, BoxType.CXCYWH, [50, 50, 100, 100], 10000),
        ([0, 0, 100, 100], BoxType.XYWH, BoxType.XYWH, [0, 0, 100, 100], 10000),
        ([0, 0, 100, 100], BoxType.XYWH, BoxType.XYXY, [0, 0, 100, 100], 10000),
        ([0, 0, 100, 100], BoxType.XYWH, BoxType.CXCYWH, [50, 50, 100, 100], 10000),
        ([50, 50, 100, 100], BoxType.CXCYWH, BoxType.CXCYWH, [50, 50, 100, 100], 10000),
        ([50, 50, 100, 100], BoxType.CXCYWH, BoxType.XYWH, [0, 0, 100, 100], 10000),
        ([50, 50, 100, 100], BoxType.CXCYWH, BoxType.XYXY, [0, 0, 100, 100], 10000),
    ],
)
def test_box(box, src_type, dst_type, converted_box, area):
    assert convert_box(box, src_type, dst_type) == converted_box
    assert get_box_area(box, src_type) == area


@pytest.mark.parametrize(
    "segmentation, box, image_size, src_type, dst_type, converted_segmentation, area",
    [
        (
            [[0, 0, 100, 0, 100, 100, 0, 100]],
            [0, 0, 100, 100],
            [100, 100],
            SegmentationType.POLYGON,
            SegmentationType.POLYGON,
            [[0, 0, 100, 0, 100, 100, 0, 100]],
            10000,
        ),
        (
            [[0, 0, 100, 0, 100, 100, 0, 100]],
            [0, 0, 100, 100],
            [100, 100],
            SegmentationType.POLYGON,
            SegmentationType.MASK,
            np.ones((100, 100), dtype=np.uint8) * 255,
            10000,
        ),
        (
            [[0, 0, 100, 0, 100, 100, 0, 100]],
            [0, 0, 100, 100],
            [100, 100],
            SegmentationType.POLYGON,
            SegmentationType.RLE,
            {
                "counts": [0, 10000],
                "size": [100, 100],
            },
            10000,
        ),
        (
            np.ones((100, 100), dtype=np.uint8) * 255,
            [0, 0, 100, 100],
            [100, 100],
            SegmentationType.MASK,
            SegmentationType.POLYGON,
            [[0, 0, 0, 100, 100, 100, 100, 0]],
            10000,
        ),
        (
            np.ones((100, 100), dtype=np.uint8) * 255,
            [0, 0, 100, 100],
            [100, 100],
            SegmentationType.MASK,
            SegmentationType.MASK,
            np.ones((100, 100), dtype=np.uint8) * 255,
            10000,
        ),
        (
            np.ones((100, 100), dtype=np.uint8) * 255,
            [0, 0, 100, 100],
            [100, 100],
            SegmentationType.MASK,
            SegmentationType.RLE,
            {
                "counts": [0, 10000],
                "size": [100, 100],
            },
            10000,
        ),
        (
            {
                "counts": [0, 10000],
                "size": [100, 100],
            },
            [0, 0, 100, 100],
            [100, 100],
            SegmentationType.RLE,
            SegmentationType.POLYGON,
            [[0, 0, 0, 100, 100, 100, 100, 0]],
            10000,
        ),
        (
            {
                "counts": [0, 10000],
                "size": [100, 100],
            },
            [0, 0, 100, 100],
            [100, 100],
            SegmentationType.RLE,
            SegmentationType.MASK,
            np.ones((100, 100), dtype=np.uint8) * 255,
            10000,
        ),
        (
            {
                "counts": [0, 10000],
                "size": [100, 100],
            },
            [0, 0, 100, 100],
            [100, 100],
            SegmentationType.RLE,
            SegmentationType.RLE,
            {
                "counts": [0, 10000],
                "size": [100, 100],
            },
            10000,
        ),
    ],
)
def test_segmentation(
    segmentation, box, image_size, src_type, dst_type, converted_segmentation, area
):
    if isinstance(converted_segmentation, np.ndarray):
        assert np.array_equal(
            convert_segmentation(segmentation, src_type, dst_type, image_size=image_size),
            converted_segmentation,
        )
    else:
        assert (
            convert_segmentation(segmentation, src_type, dst_type, image_size=image_size)
            == converted_segmentation
        )
    assert get_segmentation_area(segmentation, src_type, image_size=image_size) == area
    assert get_segmentation_box(segmentation, src_type, image_size=image_size) == box
