from typing import Optional, Union

from pydantic import field_validator

from waffle_dough.field.base_field import BaseField
from waffle_dough.math.geometry import convert_rle_to_polygon, get_polygon_area
from waffle_dough.type import TaskType


class AnnotationInfo(BaseField):
    category_id: str
    bbox: Optional[list[float]] = None
    segmentation: Optional[Union[dict, list[list[float]]]] = None
    area: Optional[float] = None
    keypoints: Optional[list[float]] = None
    num_keypoints: Optional[int] = None
    caption: Optional[str] = None
    value: Optional[float] = None
    iscrowd: Optional[int] = None
    score: Optional[float] = None
    is_prediction: Optional[bool] = None

    @field_validator("bbox")
    def check_bbox(cls, v):
        if v and len(v) != 4:
            raise ValueError("the length of bbox should be 4.")
        return v

    @field_validator("segmentation")
    def check_segmentation(cls, v):
        if v:
            for segment in v:
                if len(segment) % 2 != 0:
                    raise ValueError("the length of segmentation should be divisible by 2.")
        return v

    @field_validator("keypoints")
    def check_keypoints(cls, v):
        if v and len(v) % 3 != 0:
            raise ValueError("the length of keypoints should be divisible by 3.")
        return v

    def __init__(self, **data):
        if not hasattr(data, "task"):
            data.update({"task": TaskType.AGNOSTIC.upper()})

        super().__init__(**data)

        if isinstance(self.segmentation, dict):
            self.segmentation = convert_rle_to_polygon(self.segmentation)

        if self.area is None:
            if self.segmentation is not None:
                self.area = 0
                for polygon in self.segmentation:
                    self.area += get_polygon_area(polygon)
            elif self.bbox is not None:
                self.area = self.bbox[2] * self.bbox[3]

        if self.bbox is None:
            if self.segmentation is not None:
                xs = [x for polygon in self.segmentation for x in polygon[::2]]
                ys = [y for polygon in self.segmentation for y in polygon[1::2]]
                x1 = min(xs)
                y1 = min(ys)
                w = max(xs) - x1
                h = max(ys) - y1
                self.bbox = [x1, y1, w, h]

        if self.iscrowd is None and self.bbox is not None:
            self.iscrowd = 0

        if self.num_keypoints is None and self.keypoints is not None:
            self.num_keypoints = len(self.keypoints) // 3

        if self.score is not None:
            if self.score < 0 or self.score > 1:
                raise ValueError("score should be in [0, 1].")
            self.is_prediction = True

    @classmethod
    def classification(
        cls,
        category_id: str,
        score: float = None,
    ) -> "AnnotationInfo":
        """Classification Annotation Format

        Args:
            category_id (str): category id.
            score (float, optional): prediction score. Default to None.

        Returns:
            Annotation: annotation class
        """
        return cls(
            task=TaskType.CLASSIFICATION,
            category_id=category_id,
            score=score,
        )

    @classmethod
    def object_detection(
        cls,
        category_id: str,
        bbox: list[float],
        area: int = None,
        iscrowd: int = 0,
        score: float = None,
    ) -> "AnnotationInfo":
        """Object Detection Annotation Format

        Args:
            category_id (str): category id.
            bbox (list[float]): [x1, y1, w, h].
            area (int): bbox area.
            iscrowd (int, optional): is crowd or not. Default to 0.
            score (float, optional): prediction score. Default to None.

        Returns:
            Annotation: annotation class
        """
        return cls(
            task=TaskType.OBJECT_DETECTION,
            category_id=category_id,
            bbox=bbox,
            area=area,
            iscrowd=iscrowd,
            score=score,
        )

    @classmethod
    def semantic_segmentation(
        cls,
        category_id: str,
        segmentation: Union[list[list[float]], dict],
        bbox: list[float] = None,
        area: int = None,
        iscrowd: int = 0,
        score: float = None,
    ) -> "AnnotationInfo":
        """Segmentation Annotation Format

        Args:
            category_id (str): category id.
            bbox (list[float]): [x1, y1, w, h].
            segmentation (Union[list[list[float]], dict]): [[x1, y1, x2, y2, x3, y3, ...], [polygon]] or RLE.
            area (int): segmentation segmentation area.
            iscrowd (int, optional): is crowd or not. Default to 0.
            score (float, optional): prediction score. Default to None.

        Returns:
            Annotation: annotation class
        """
        return cls(
            task=TaskType.SEMANTIC_SEGMENTATION,
            category_id=category_id,
            bbox=bbox,
            segmentation=segmentation,
            area=area,
            iscrowd=iscrowd,
            score=score,
        )

    @classmethod
    def instance_segmentation(
        cls,
        category_id: str,
        segmentation: Union[list[list[float]], dict],
        bbox: list[float] = None,
        area: int = None,
        iscrowd: int = 0,
        score: float = None,
    ) -> "AnnotationInfo":
        """Instance Annotation Format

        Args:
            category_id (str): category id.
            bbox (list[float]): [x1, y1, w, h].
            segmentation (Union[list[list[float]], dict]): [[x1, y1, x2, y2, x3, y3, ...], [polygon]] or RLE.
            area (int): segmentation segmentation area.
            iscrowd (int, optional): is crowd or not. Default to 0.
            score (float, optional): prediction score. Default to None.

        Returns:
            Annotation: annotation class
        """
        return cls(
            task=TaskType.INSTANCE_SEGMENTATION,
            category_id=category_id,
            bbox=bbox,
            segmentation=segmentation,
            area=area,
            iscrowd=iscrowd,
            score=score,
        )

    @classmethod
    def keypoint_detection(
        cls,
        category_id: str,
        keypoints: list[float],
        bbox: list[float],
        num_keypoints: int = None,
        area: int = None,
        segmentation: list[list[float]] = None,
        iscrowd: int = 0,
        score: list[float] = None,
    ) -> "AnnotationInfo":
        """Keypoint Detection Annotation Format

        Args:
            category_id (str): category id.
            bbox (list[float]): [x1, y1, w, h].
            keypoints (list[float]):
                [x1, y1, v1(visible flag), x2, y2, v2(visible flag), ...].
                visible flag is one of [0(Not labeled), 1(Labeled but not visible), 2(labeled and visible)]
            num_keypoints: number of labeled keypoints
            area (int): segmentation segmentation or bbox area.
            segmentation (list[list[float]], optional): [[x1, y1, x2, y2, x3, y3, ...], [polygon]].
            iscrowd (int, optional): is crowd or not. Default to 0.
            score (list[float], optional): prediction scores. Default to None.

        Returns:
            Annotation: annotation class
        """
        return cls(
            task=TaskType.KEYPOINT_DETECTION,
            category_id=category_id,
            bbox=bbox,
            keypoints=keypoints,
            num_keypoints=num_keypoints,
            segmentation=segmentation,
            area=area,
            iscrowd=iscrowd,
            score=score,
        )

    @classmethod
    def regression(cls, category_id: str, value: float) -> "AnnotationInfo":
        """Regression Annotation Format

        Args:
            category_id (str): category id.
            value (float): regression value.

        Returns:
            Annotation: annotation class
        """
        return cls(
            task=TaskType.REGRESSION,
            category_id=category_id,
            value=value,
        )

    @classmethod
    def text_recognition(
        cls,
        category_id: str,
        caption: str,
        score: float = None,
    ) -> "AnnotationInfo":
        """Text Recognition Annotation Format

        Args:
            category_id (str): category id.
            caption (str): string.
            score (float, optional): prediction score. Default to None.

        Returns:
            Annotation: annotation class
        """
        return cls(
            task=TaskType.TEXT_RECOGNITION,
            category_id=category_id,
            caption=caption,
            score=score,
        )
