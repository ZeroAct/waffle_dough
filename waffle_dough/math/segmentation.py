from itertools import groupby
from typing import Any, Union

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from shapely.geometry import Polygon as ShapelyPolygon

from waffle_dough.math.box import convert_box
from waffle_dough.type.annotation_type import BoxType, SegmentationType


def mask_to_polygon(mask: np.ndarray) -> list[list]:
    """Convert mask to polygon.
    Args:
        mask (np.ndarray): mask image.

    Returns:
        list[list]: polygon coordinates. [[x1, y1, x2, y2, ...], ...]
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []

    for contour in contours:
        polygon = []
        contour = contour.reshape(-1, 2)
        for point in contour:
            x, y = point
            if x != 0:
                x += 1
            if y != 0:
                y += 1
            polygon.extend([x, y])
        polygons.append(polygon)
    return polygons


def mask_to_rle(mask: np.ndarray) -> dict:
    """Convert mask to rle.
    Args:
        mask (np.ndarray): mask image.

    Returns:
        dict: rle.
    """
    rle = {"counts": [], "size": list(mask.shape)}
    counts = rle.get("counts")
    for i, (value, elements) in enumerate(groupby(mask.ravel(order="F"))):
        if i == 0 and value != 0:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


def polygon_to_mask(
    polygons: list[list[float]], image_size: tuple, fill_value: int = 255
) -> np.ndarray:
    """Convert polygon to mask.
    Args:
        polygons (list[float]): polygons coordinates. [[x1, y1, x2, y2, x3, y3, ...], []]
        image_size (tuple): image size. (height, width)
        fill_value (int): fill value for mask.

    Returns:
        np.ndarray: mask image.
    """
    mask = np.zeros(image_size, dtype=np.uint8)
    for polygon in polygons:
        points = np.array(polygon).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [points], fill_value)
    return mask


def polygon_to_rle(polygons: list[float], image_size: tuple) -> dict:
    """Convert polygon to rle.
    Args:
        polygons (list[float]): polygons coordinates. [[x1, y1, x2, y2, x3, y3, ...], []]
        image_size (tuple): image size. (height, width)

    Returns:
        dict: rle.
    """
    mask = polygon_to_mask(polygons, image_size)
    return mask_to_rle(mask)


def rle_to_mask(rle: dict, fill_value: int = 255) -> np.ndarray:
    """Convert rle to mask.
    Args:
        rle (dict): rle.
        fill_value (int): fill value for mask.

    Returns:
        np.ndarray: mask image.
    """
    mask = mask_utils.frPyObjects(rle, rle["size"][0], rle["size"][1])  # height, width
    return mask_utils.decode(mask) * fill_value


def rle_to_polygon(rle: dict) -> list[float]:
    """Convert rle to polygon.
    Args:
        rle (dict): rle.

    Returns:
        list[float]: polygon coordinates. [x1, y1, x2, y2, x3, y3, ...]
    """
    mask = rle_to_mask(rle)
    return mask_to_polygon(mask)


def convert_segmentation(
    segmentation: Any,
    src_type: Union[str, SegmentationType],
    dst_type: Union[str, SegmentationType],
    image_size: tuple[int, int] = None,
) -> Union[list[list[float]], dict, np.ndarray]:
    """Convert segmentation from src_type to dst_type.
    Args:
        segmentation (list[list[float]]): segmentation
        src_type (Union[str, SegmentationType]): source type
        dst_type (Union[str, SegmentationType]): destination type
        image_size (Optional(tuple[int, int])): image size for polygon type. (width, height)

    Returns:
        Union[list[list[float]], dict, np.ndarray]: converted segmentation
    """

    if src_type == dst_type:
        return segmentation

    if src_type == SegmentationType.POLYGON:
        if image_size is None:
            raise ValueError(
                "image_size must be provided when converting from polygon to other types."
            )
        if dst_type == SegmentationType.MASK:
            return polygon_to_mask(segmentation, image_size)
        elif dst_type == SegmentationType.RLE:
            return polygon_to_rle(segmentation, image_size)
    elif src_type == SegmentationType.MASK:
        if dst_type == SegmentationType.POLYGON:
            return mask_to_polygon(segmentation)
        elif dst_type == SegmentationType.RLE:
            return mask_to_rle(segmentation)
    elif src_type == SegmentationType.RLE:
        if dst_type == SegmentationType.POLYGON:
            return rle_to_polygon(segmentation)
        elif dst_type == SegmentationType.MASK:
            return rle_to_mask(segmentation)
    else:
        raise NotImplementedError(f"Cannot convert segmentation from {src_type} to {dst_type}")


def get_segmentation_area(
    segmentation: list[list[float]],
    segmentation_type: Union[str, SegmentationType],
    image_size: tuple[int, int] = None,
) -> float:
    """Get segmentation area.
    Args:
        segmentation (list[list[float]]): segmentation
        segmentation_type (Union[str, SegmentationType]): segmentation type
        image_size (Optional(tuple[int, int])): image size for polygon type. (width, height)

    Returns:
        float: segmentation area
    """
    polygons = convert_segmentation(
        segmentation, segmentation_type, SegmentationType.POLYGON, image_size=image_size
    )
    area = 0
    for polygon in polygons:
        polygon = ShapelyPolygon(np.array(polygon).reshape(-1, 2))
        area += polygon.area
    return area


def get_segmentation_box(
    segmentation: list[list[float]],
    segmentation_type: Union[str, SegmentationType],
    image_size: tuple[int, int] = None,
    box_type: Union[str, BoxType] = BoxType.XYWH,
) -> list[float]:
    """Get segmentation box.
    Args:
        segmentation (list[list[float]]): segmentation
        segmentation_type (Union[str, SegmentationType]): segmentation type
        image_size (Optional(tuple[int, int])): image size for polygon type. (width, height)
        box_type (Union[str, BoxType]): box type. Default to BoxType.XYWH.

    Returns:
        list[float]: segmentation bbox [x1, y1, x2, y2]
    """
    polygons = convert_segmentation(
        segmentation, segmentation_type, SegmentationType.POLYGON, image_size=image_size
    )
    x1, y1, x2, y2 = np.inf, np.inf, -np.inf, -np.inf
    for polygon in polygons:
        polygon = np.array(polygon).reshape(-1, 2)
        x1 = min(x1, polygon[:, 0].min())
        y1 = min(y1, polygon[:, 1].min())
        x2 = max(x2, polygon[:, 0].max())
        y2 = max(y2, polygon[:, 1].max())
    return convert_box([x1, y1, x2, y2], BoxType.XYXY, box_type)
