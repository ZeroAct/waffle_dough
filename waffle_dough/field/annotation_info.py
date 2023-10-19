from typing import Union

from waffle_utils.validator import setter_type_validator

from waffle_dough.type import TaskType
from waffle_dough.util.math.geometry import convert_rle_to_polygon, get_polygon_area

from .base_field import BaseField


class AnnotationInfo(BaseField):
    def __init__(
        self,
        task: Union[str, TaskType],
        annotation_id: int,
        image_id: int,
        category_id: int = None,
        bbox: list[float] = None,
        segmentation: list[list[float]] = None,
        area: float = None,
        keypoints: list[float] = None,
        num_keypoints: int = None,
        caption: str = None,
        value: float = None,
        iscrowd: int = None,
        score: Union[float, list[float]] = None,
    ):

        self.task = task
        self.annotation_id = annotation_id
        self.image_id = image_id
        self.category_id = category_id
        self.bbox = bbox
        self.segmentation = segmentation
        self.area = area
        self.keypoints = keypoints
        self.num_keypoints = num_keypoints
        self.caption = caption
        self.value = value
        self.iscrowd = iscrowd
        self.score = score

    # properties
    @property
    def annotation_id(self) -> int:
        return self.__annotation_id

    @annotation_id.setter
    @setter_type_validator(int)
    def annotation_id(self, v):
        if v and v < 1:
            raise ValueError("id should be greater than 0.")
        self.__annotation_id = v

    @property
    def image_id(self) -> int:
        return self.__image_id

    @image_id.setter
    @setter_type_validator(int)
    def image_id(self, v):
        if v and v < 1:
            raise ValueError("id should be greater than 0.")
        self.__image_id = v

    @property
    def category_id(self) -> int:
        return self.__category_id

    @category_id.setter
    @setter_type_validator(int)
    def category_id(self, v):
        if v and v < 1:
            raise ValueError("id should be greater than 0.")
        self.__category_id = v

    @property
    def bbox(self) -> list[float]:
        return self.__bbox

    @bbox.setter
    @setter_type_validator(list)
    def bbox(self, v):
        if v and len(v) != 4:
            raise ValueError("the length of bbox should be 4.")
        self.__bbox = v

    @property
    def segmentation(self) -> list[list[float]]:
        return self.__segmentation

    @segmentation.setter
    def segmentation(self, v):
        if v:
            for segment in v:
                if len(segment) % 2 != 0:
                    raise ValueError("the length of segmentation should be divisible by 2.")

        self.__segmentation = v

    @property
    def area(self) -> float:
        return self.__area

    @area.setter
    @setter_type_validator(float, strict=False)
    def area(self, v):
        self.__area = v

    @property
    def keypoints(self) -> list[float]:
        return self.__keypoints

    @keypoints.setter
    @setter_type_validator(list)
    def keypoints(self, v):
        if v and len(v) % 3 != 0 and len(v) < 2:
            raise ValueError("the length of keypoints should be at least 2 and divisible by 3.")
        self.__keypoints = v

    @property
    def num_keypoints(self) -> int:
        return self.__num_keypoints

    @num_keypoints.setter
    @setter_type_validator(int)
    def num_keypoints(self, v):
        self.__num_keypoints = v

    @property
    def caption(self) -> str:
        return self.__caption

    @caption.setter
    @setter_type_validator(str)
    def caption(self, v):
        self.__caption = v

    @property
    def value(self) -> float:
        return self.__value

    @value.setter
    @setter_type_validator(float, strict=False)
    def value(self, v):
        self.__value = v

    @property
    def iscrowd(self) -> int:
        return self.__iscrowd

    @iscrowd.setter
    @setter_type_validator(int, strict=False)
    def iscrowd(self, v):
        self.__iscrowd = v

    @property
    def score(self) -> float:
        return self.__score

    @score.setter
    @setter_type_validator(float, strict=False)
    def score(self, v):
        self.__score = v

    @property
    def task(self) -> str:
        return self.__task

    @task.setter
    @setter_type_validator(TaskType, strict=False)
    def task(self, v):
        if v is not None and v not in list(TaskType):
            raise ValueError(f"Invalid task type: {v}" f"Available task types: {list(TaskType)}")
        self.__task = v.upper()

    def __eq__(self, other):
        if not isinstance(other, AnnotationInfo):
            return False

        if self.task == TaskType.CLASSIFICATION:
            return self.category_id == other.category_id
        elif self.task == TaskType.OBJECT_DETECTION:
            return (
                self.category_id == other.category_id
                and self.bbox == other.bbox
                and self.iscrowd == other.iscrowd
            )
        elif self.task == TaskType.SEMANTIC_SEGMENTATION:
            return (
                self.category_id == other.category_id
                and self.segmentation == other.segmentation
                and self.iscrowd == other.iscrowd
            )
        elif self.task == TaskType.INSTANCE_SEGMENTATION:
            return (
                self.category_id == other.category_id
                and self.segmentation == other.segmentation
                and self.iscrowd == other.iscrowd
            )
        elif self.task == TaskType.KEYPOINT_DETECTION:
            return (
                self.category_id == other.category_id
                and self.keypoints == other.keypoints
                and self.num_keypoints == other.num_keypoints
                and self.iscrowd == other.iscrowd
            )
        elif self.task == TaskType.TEXT_RECOGNITION:
            return self.caption == other.caption
        elif self.task == TaskType.REGRESSION:
            return self.value == other.value
        else:
            raise NotImplementedError(f"Task type {self.task} is not supported.")

    # factories
    @classmethod
    def new(
        cls,
        task: Union[str, TaskType],
        *args,
        **kwargs,
    ) -> "AnnotationInfo":
        """Annotation Format

        Args:
            task (Union[str, TaskType]): task type.
            *args: args.
            **kwargs: kwargs.

        Returns:
            Annotation: annotation class
        """
        if task not in list(TaskType):
            raise ValueError(f"Invalid task type: {task}" f"Available task types: {list(TaskType)}")

        return getattr(cls, task.lower())(*args, **kwargs)

    @classmethod
    def classification(
        cls,
        annotation_id: int,
        image_id: int,
        category_id: int,
        score: float = None,
    ) -> "AnnotationInfo":
        """Classification Annotation Format

        Args:
            annotation_id (int): annotaion id. natural number.
            image_id (int): image id. natural number.
            category_id (int): category id. natural number.
            score (float, optional): prediction score. Default to None.

        Returns:
            Annotation: annotation class
        """
        return cls(
            task=TaskType.CLASSIFICATION,
            annotation_id=annotation_id,
            image_id=image_id,
            category_id=category_id,
            score=score,
        )

    @classmethod
    def object_detection(
        cls,
        annotation_id: int,
        image_id: int,
        category_id: int,
        bbox: list[float],
        area: int = None,
        iscrowd: int = 0,
        score: float = None,
    ) -> "AnnotationInfo":
        """Object Detection Annotation Format

        Args:
            annotation_id (int): annotaion id. natural number.
            image_id (int): image id. natural number.
            category_id (int): category id. natural number.
            bbox (list[float]): [x1, y1, w, h].
            area (int): bbox area.
            iscrowd (int, optional): is crowd or not. Default to 0.
            score (float, optional): prediction score. Default to None.

        Returns:
            Annotation: annotation class
        """

        if area is None:
            area = bbox[2] * bbox[3]

        return cls(
            task=TaskType.OBJECT_DETECTION,
            annotation_id=annotation_id,
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
        annotation_id: int,
        image_id: int,
        category_id: int,
        segmentation: Union[list[list[float]], dict],
        bbox: list[float] = None,
        area: int = None,
        iscrowd: int = 0,
        score: float = None,
    ) -> "AnnotationInfo":
        """Segmentation Annotation Format

        Args:
            annotation_id (int): annotaion id. natural number.
            image_id (int): image id. natural number.
            category_id (int): category id. natural number.
            bbox (list[float]): [x1, y1, w, h].
            segmentation (Union[list[list[float]], dict]): [[x1, y1, x2, y2, x3, y3, ...], [polygon]] or RLE.
            area (int): segmentation segmentation area.
            iscrowd (int, optional): is crowd or not. Default to 0.
            score (float, optional): prediction score. Default to None.

        Returns:
            Annotation: annotation class
        """

        if isinstance(segmentation, dict):
            segmentation = convert_rle_to_polygon(segmentation)

        if bbox is None:
            xs = [x for polygon in segmentation for x in polygon[::2]]
            ys = [y for polygon in segmentation for y in polygon[1::2]]
            x1 = min(xs)
            y1 = min(ys)
            w = max(xs) - x1
            h = max(ys) - y1
            bbox = [x1, y1, w, h]

        if area is None:
            area = 0
            for polygon in segmentation:
                area += get_polygon_area(polygon)

        return cls(
            task=TaskType.SEMANTIC_SEGMENTATION,
            annotation_id=annotation_id,
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
        annotation_id: int,
        image_id: int,
        category_id: int,
        segmentation: Union[list[list[float]], dict],
        bbox: list[float] = None,
        area: int = None,
        iscrowd: int = 0,
        score: float = None,
    ) -> "AnnotationInfo":
        """Instance Annotation Format

        Args:
            annotation_id (int): annotaion id. natural number.
            image_id (int): image id. natural number.
            category_id (int): category id. natural number.
            bbox (list[float]): [x1, y1, w, h].
            segmentation (Union[list[list[float]], dict]): [[x1, y1, x2, y2, x3, y3, ...], [polygon]] or RLE.
            area (int): segmentation segmentation area.
            iscrowd (int, optional): is crowd or not. Default to 0.
            score (float, optional): prediction score. Default to None.

        Returns:
            Annotation: annotation class
        """

        if isinstance(segmentation, dict):
            segmentation = convert_rle_to_polygon(segmentation)

        if bbox is None:
            xs = [x for polygon in segmentation for x in polygon[::2]]
            ys = [y for polygon in segmentation for y in polygon[1::2]]
            x1 = min(xs)
            y1 = min(ys)
            w = max(xs) - x1
            h = max(ys) - y1
            bbox = [x1, y1, w, h]

        if area is None:
            area = 0
            for polygon in segmentation:
                area += get_polygon_area(polygon)

        return cls(
            task=TaskType.INSTANCE_SEGMENTATION,
            annotation_id=annotation_id,
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
        annotation_id: int,
        image_id: int,
        category_id: int,
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
            annotation_id (int): annotaion id. natural number.
            image_id (int): image id. natural number.
            category_id (int): category id. natural number.
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

        if num_keypoints is None and keypoints is not None:
            num_keypoints = len(keypoints) // 3

        if area is None:
            area = bbox[2] * bbox[3]

        return cls(
            task=TaskType.KEYPOINT_DETECTION,
            annotation_id=annotation_id,
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
    def regression(cls, annotation_id: int, image_id: int, value: float) -> "AnnotationInfo":
        """Regression Annotation Format

        Args:
            annotation_id (int): annotaion id. natural number.
            image_id (int): image id. natural number.
            value (float): regression value.

        Returns:
            Annotation: annotation class
        """
        return cls(
            task=TaskType.REGRESSION,
            annotation_id=annotation_id,
            image_id=image_id,
            value=value,
        )

    @classmethod
    def text_recognition(
        cls,
        annotation_id: int,
        image_id: int,
        caption: str,
        score: float = None,
    ) -> "AnnotationInfo":
        """Text Recognition Annotation Format

        Args:
            annotation_id (int): annotaion id. natural number.
            image_id (int): image id. natural number.
            caption (str): string.
            score (float, optional): prediction score. Default to None.

        Returns:
            Annotation: annotation class
        """
        return cls(
            task=TaskType.TEXT_RECOGNITION,
            annotation_id=annotation_id,
            image_id=image_id,
            caption=caption,
            score=score,
        )

    def to_dict(self) -> dict:
        """Get Dictionary of Annotation Data

        Returns:
            dict: annotation dictionary.
        """

        ann = {}
        if self.annotation_id is not None:
            ann["annotation_id"] = self.annotation_id
        if self.image_id is not None:
            ann["image_id"] = self.image_id
        if self.category_id is not None:
            ann["category_id"] = self.category_id
        if self.bbox is not None:
            ann["bbox"] = self.bbox
        if self.segmentation is not None:
            ann["segmentation"] = self.segmentation
        if self.area is not None:
            ann["area"] = self.area
        if self.keypoints is not None:
            ann["keypoints"] = self.keypoints
        if self.caption is not None:
            ann["caption"] = self.caption
        if self.value is not None:
            ann["value"] = self.value
        if self.iscrowd is not None:
            ann["iscrowd"] = self.iscrowd
        if self.score is not None:
            ann["score"] = self.score

        return ann

    def is_prediction(self):
        return not self.score is None
