import logging
import uuid
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from waffle_utils.file.network import get_file_from_url

from waffle_dough.exception import BaseException
from waffle_dough.field import AnnotationInfo, CategoryInfo, ImageInfo
from waffle_dough.image import io
from waffle_dough.type import TaskType

FONT_URL = "https://raw.githubusercontent.com/snuailab/assets/main/waffle/fonts/gulim.ttc"
FONT_NAME = "gulim.ttc"
BASE_FONT_SIZE = 15


# random colors with 1000 categories
colors = [
    [255, 30, 30],
    [30, 255, 30],
    [30, 30, 255],
    [255, 255, 30],
    [255, 30, 255],
    [30, 255, 255],
    [200, 100, 20],
    [20, 200, 100],
    [100, 20, 200],
    [160, 0, 220],
    [220, 160, 0],
    [0, 220, 160],
] + np.random.randint(0, 255, (1000, 3), dtype="uint8").tolist()


def _get_font(scale: float = 1.0) -> ImageFont.ImageFont:
    if not Path(FONT_NAME).exists():
        get_file_from_url(FONT_URL, FONT_NAME, True)
    return ImageFont.truetype(FONT_NAME, int(BASE_FONT_SIZE * scale))


def _convert_to_pil(image: np.ndarray) -> Image.Image:
    if image.dtype == np.uint8:
        return Image.fromarray(image)
    elif image.dtype == np.float32:
        return Image.fromarray((image * 255).astype(np.uint8))
    else:
        raise BaseException("image dtype must be uint8 or float32")


def _convert_to_numpy(image: Image.Image) -> np.ndarray:
    return np.array(image)


def visualize_classification(
    image: np.ndarray,
    annotations: list[AnnotationInfo],
    category_dict: dict[CategoryInfo],
    image_info: ImageInfo,
    color_map: dict[str, list[int]],
    scale: float = 1.0,
    opacity: float = 1.0,
) -> np.ndarray:

    pil_image = _convert_to_pil(image)
    draw = ImageDraw.Draw(pil_image, "RGBA")
    font = _get_font(scale)

    for i, annotation in enumerate(annotations):
        category = category_dict[annotation.category_id]
        color = tuple(color_map[annotation.category_id])
        text = f"{category.name}" + (f": {annotation.score}" if annotation.score is not None else "")

        text_box = draw.textbbox((0, 0 + 20 * i), text, font=font)
        draw.rectangle(
            text_box, fill=color + (int(255 * opacity),), outline=color, width=round(2 * scale)
        )
        draw.text(text_box[:2], text, (255, 255, 255), font=font)

    return _convert_to_numpy(pil_image)


def visualize_regression(
    image: np.ndarray,
    annotations: list[AnnotationInfo],
    category_dict: dict[CategoryInfo],
    image_info: ImageInfo,
    color_map: dict[str, list[int]],
    scale: float = 1.0,
    opacity: float = 1.0,
) -> np.ndarray:

    pil_image = _convert_to_pil(image)
    draw = ImageDraw.Draw(pil_image, "RGBA")
    font = _get_font(scale)

    for i, annotation in enumerate(annotations):
        category = category_dict[annotation.category_id]
        color = tuple(color_map[annotation.category_id])
        text = f"{category.name}: {annotation.value}"

        text_box = draw.textbbox((0, 0 + 20 * i), text, font=font)
        draw.rectangle(
            text_box, fill=color + (int(255 * opacity),), outline=color, width=round(2 * scale)
        )
        draw.text(text_box[:2], text, (255, 255, 255), font=font)

    return _convert_to_numpy(pil_image)


def visualize_object_detection(
    image: np.ndarray,
    annotations: list[AnnotationInfo],
    category_dict: dict[CategoryInfo],
    image_info: ImageInfo,
    color_map: dict[str, list[int]],
    scale: float = 1.0,
    opacity: float = 0.2,
) -> np.ndarray:

    pil_image = _convert_to_pil(image)
    draw = ImageDraw.Draw(pil_image, "RGBA")
    font = _get_font(scale)

    W, H = image_info.width, image_info.height

    for i, annotation in enumerate(annotations):
        category = category_dict[annotation.category_id]
        color = tuple(color_map[annotation.category_id])
        text = f"{category.name}" + (f": {annotation.score}" if annotation.score is not None else "")

        x1, y1, w, h = annotation.bbox
        x2, y2 = x1 + w, y1 + h
        x1, y1, x2, y2 = round(x1 * W), round(y1 * H), round(x2 * W), round(y2 * H)

        draw.rectangle(
            (x1, y1, x2, y2),
            fill=color + (int(127 * opacity),),
            outline=color,
            width=round(2 * scale),
        )

        text_box = draw.textbbox((x1, y1), text, font=font)
        th = text_box[3] - text_box[1]
        th = round(th * 1.25)
        pad = round(th * 0.1)
        text_box = (text_box[0], text_box[1] - th - pad, text_box[2], text_box[3] - th + pad)
        draw.rectangle(text_box, fill=color, outline=color, width=round(2 * scale))
        draw.text(text_box[:2], text, (255, 255, 255), font=font)

    return _convert_to_numpy(pil_image)


def visualize(
    image: np.ndarray,
    annotations: Union[AnnotationInfo, list[AnnotationInfo], dict, list[dict]],
    category_dict: dict[str, CategoryInfo],
    image_info: Union[ImageInfo, dict] = None,
    task: Union[TaskType, str] = None,
    color_map: dict[str, list[int]] = None,
    opacity: float = None,
) -> np.ndarray:
    """
    Visualize image with annotations and categories

    Args:
        image (np.ndarray): image to visualize
        image_infos (ImageInfo): image info to visualize
        annotations (Union[AnnotationInfo, list[AnnotationInfo]]): annotations to visualize
        category_dict (dict[str, CategoryInfo]): category dict to visualize
        task (Union[TaskType, str], optional): task type. Defaults to None.
        color_map (dict[str, list[int]], optional): color map that has category id as key and color as value. Defaults to None.
        opacity (float, optional): opacity of visualized image. Defaults to None.

    Returns:
        np.ndarray: visualized image
    """

    result = image.copy()

    # parse arguments
    ## get image info
    if image_info is not None:
        image_info = ImageInfo.from_dict(image_info) if isinstance(image_info, dict) else image_info
    else:
        image_info = ImageInfo(
            ext=".jpg",
            width=image.shape[1],
            height=image.shape[0],
            original_file_name=f"{uuid.uuid4()}.jpg",
        )

    ## get annotations
    annotations = annotations if isinstance(annotations, list) else [annotations]
    annotations = [
        AnnotationInfo.from_dict(annotation) if isinstance(annotation, dict) else annotation
        for annotation in annotations
    ]

    if len(annotations) == 0:
        return result

    ## get task type
    if task is None:
        task = TaskType.from_str(annotations[0].task)
    elif isinstance(task, str):
        task = TaskType.from_str(task)

    # visualize
    ## get color map
    if color_map is None:
        color_map = {}
        for i, category_id in enumerate(category_dict):
            color_map[category_id] = colors[i]

    ## get adaptive font scale
    scale = min(image.shape[0], image.shape[1]) / 256

    if task == TaskType.CLASSIFICATION:
        result = visualize_classification(
            result,
            annotations,
            category_dict,
            image_info,
            color_map,
            scale,
            **{"opacity": opacity} if opacity is not None else {},
        )
    elif task == TaskType.OBJECT_DETECTION:
        result = visualize_object_detection(
            result,
            annotations,
            category_dict,
            image_info,
            color_map,
            scale,
            **{"opacity": opacity} if opacity is not None else {},
        )
    elif task == TaskType.REGRESSION:
        result = visualize_regression(
            result,
            annotations,
            category_dict,
            image_info,
            color_map,
            scale,
            **{"opacity": opacity} if opacity is not None else {},
        )

    return result
