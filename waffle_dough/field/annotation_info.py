from typing import ClassVar, Optional, Union

from pydantic import Field, field_validator

from waffle_dough.field.base_field import BaseField
from waffle_dough.math.geometry import convert_rle_to_polygon, get_polygon_area
from waffle_dough.type import TaskType


class AnnotationInfo(BaseField):
    image_id: str = Field(...)
    category_id: str = Field(None)
    bbox: Optional[list[float]] = Field(None)
    segmentation: Optional[Union[dict, list[list[float]]]] = Field(None)
    area: Optional[float] = Field(None)
    keypoints: Optional[list[float]] = Field(None)
    num_keypoints: Optional[int] = Field(None)
    caption: Optional[str] = Field(None)
    value: Optional[float] = Field(None)
    iscrowd: Optional[int] = Field(None)
    score: Optional[float] = Field(None)
    is_prediction: Optional[bool] = Field(None)

    extra_required_fields: ClassVar[dict[TaskType, list[str]]] = {
        TaskType.CLASSIFICATION: ["image_id", "category_id"],
        TaskType.OBJECT_DETECTION: ["image_id", "category_id", "bbox"],
        TaskType.SEMANTIC_SEGMENTATION: ["image_id", "category_id", "segmentation"],
        TaskType.INSTANCE_SEGMENTATION: ["image_id", "category_id", "segmentation"],
        TaskType.KEYPOINT_DETECTION: ["image_id", "category_id", "keypoints"],
        TaskType.TEXT_RECOGNITION: ["image_id", "caption"],
        TaskType.REGRESSION: ["image_id", "category_id", "value"],
    }

    @field_validator("bbox")
    def check_bbox(cls, v):
        if v:
            if len(v) != 4:
                raise ValueError("the length of bbox should be 4.")
            if v[2] <= 0 or v[3] <= 0:
                raise ValueError("the width and height of bbox should be greater than 0.")
        return v

    @field_validator("segmentation")
    def check_segmentation(cls, v):
        if v:
            if isinstance(v, dict):
                if "counts" not in v or "size" not in v:
                    raise ValueError(f"the segmentation should have 'counts' and 'size'. Given {v}")
                v = convert_rle_to_polygon(v)

            for segment in v:
                if len(segment) % 2 != 0:
                    raise ValueError("the length of segmentation should be divisible by 2.")
                if len(segment) < 6:
                    raise ValueError(
                        "the length of segmentation should be greater than or equal to 6."
                    )
        return v

    @field_validator("area")
    def check_area(cls, v):
        if v and v < 0:
            raise ValueError("area should be greater than or equal to 0.")
        return v

    @field_validator("keypoints")
    def check_keypoints(cls, v):
        if v and len(v) % 3 != 0:
            raise ValueError("the length of keypoints should be divisible by 3.")
        return v

    @field_validator("num_keypoints")
    def check_num_keypoints(cls, v):
        if v and v < 0:
            raise ValueError("num_keypoints should be greater than or equal to 0.")
        return v

    @field_validator("iscrowd")
    def check_iscrowd(cls, v):
        if v and v not in [0, 1]:
            raise ValueError("iscrowd should be 0 or 1.")
        return v

    @field_validator("score")
    def check_score(cls, v):
        if v and (v < 0 or v > 1):
            raise ValueError("score should be in [0, 1].")
        return v

    @field_validator("is_prediction")
    def check_is_prediction(cls, v):
        if v and not isinstance(v, bool):
            raise ValueError("is_prediction should be bool.")
        return v

    @field_validator("value")
    def check_value(cls, v):
        if v and not isinstance(v, (int, float)):
            raise ValueError("value should be int or float.")
        return v

    @field_validator("caption")
    def check_caption(cls, v):
        if v and not isinstance(v, str):
            raise ValueError("caption should be str.")
        return v

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_default_values()

    def set_default_values(self):
        # init default values
        if self.bbox is None:
            if self.segmentation is not None:
                xs = [x for polygon in self.segmentation for x in polygon[::2]]
                ys = [y for polygon in self.segmentation for y in polygon[1::2]]
                x1 = min(xs)
                y1 = min(ys)
                w = max(xs) - x1
                h = max(ys) - y1
                self.bbox = [x1, y1, w, h]

        if self.area is None:
            if self.segmentation is not None:
                self.area = 0
                for polygon in self.segmentation:
                    self.area += get_polygon_area(polygon)
            elif self.bbox is not None:
                self.area = self.bbox[2] * self.bbox[3]

        if self.iscrowd is None and self.bbox is not None:
            self.iscrowd = 0

        if self.num_keypoints is None and self.keypoints is not None:
            self.num_keypoints = len(self.keypoints) // 3

        if self.score is not None:
            self.is_prediction = True

    @classmethod
    def classification(
        cls,
        image_id: str,
        category_id: str,
        score: float = None,
    ) -> "AnnotationInfo":
        """Classification Annotation Format

        Args:
            image_id (str): image id.
            category_id (str): category id.
            score (float, optional): prediction score. Default to None.

        Returns:
            Annotation: annotation class
        """
        return cls(
            task=TaskType.CLASSIFICATION,
            image_id=image_id,
            category_id=category_id,
            score=score,
        )

    @classmethod
    def object_detection(
        cls,
        image_id: str,
        category_id: str,
        bbox: list[float],
        area: int = None,
        iscrowd: int = 0,
        score: float = None,
    ) -> "AnnotationInfo":
        """Object Detection Annotation Format

        Args:
            image_id (str): image id.
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
            image_id=image_id,
            category_id=category_id,
            bbox=bbox,
            area=area,
            iscrowd=iscrowd,
            score=score,
        )

    @classmethod
    def semantic_segmentation(
        cls,
        image_id: str,
        category_id: str,
        segmentation: Union[list[list[float]], dict],
        bbox: list[float] = None,
        area: int = None,
        iscrowd: int = 0,
        score: float = None,
    ) -> "AnnotationInfo":
        """Segmentation Annotation Format

        Args:
            image_id (str): image id.
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
            image_id=image_id,
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
        image_id: str,
        category_id: str,
        segmentation: Union[list[list[float]], dict],
        bbox: list[float] = None,
        area: int = None,
        iscrowd: int = 0,
        score: float = None,
    ) -> "AnnotationInfo":
        """Instance Annotation Format

        Args:
            image_id (str): image id.
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
            image_id=image_id,
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
        image_id: str,
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
            image_id (str): image id.
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
            image_id=image_id,
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
    def regression(cls, image_id: str, category_id: str, value: float) -> "AnnotationInfo":
        """Regression Annotation Format

        Args:
            image_id (str): image id.
            category_id (str): category id.
            value (float): regression value.

        Returns:
            Annotation: annotation class
        """
        return cls(
            task=TaskType.REGRESSION,
            image_id=image_id,
            category_id=category_id,
            value=value,
        )

    @classmethod
    def text_recognition(
        cls,
        image_id: str,
        category_id: str,
        caption: str,
        score: float = None,
    ) -> "AnnotationInfo":
        """Text Recognition Annotation Format

        Args:
            image_id (str): image id.
            category_id (str): category id.
            caption (str): string.
            score (float, optional): prediction score. Default to None.

        Returns:
            Annotation: annotation class
        """
        return cls(
            task=TaskType.TEXT_RECOGNITION,
            image_id=image_id,
            category_id=category_id,
            caption=caption,
            score=score,
        )
