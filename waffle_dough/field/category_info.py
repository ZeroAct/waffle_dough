from typing import Union

from waffle_utils.validator import setter_type_validator

from waffle_dough.type import TaskType

from .base_field import BaseField


class CategoryInfo(BaseField):
    def __init__(
        self,
        task: Union[str, TaskType],
        category_id: int,
        name: str,
        supercategory: str = None,
        keypoints: list[str] = None,
        skeleton: list[list[int]] = None,
    ):

        self.task = task
        self.category_id = category_id
        self.supercategory = supercategory
        self.name = name
        self.keypoints = keypoints
        self.skeleton = skeleton

    # properties
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
    def supercategory(self) -> str:
        return self.__supercategory

    @supercategory.setter
    @setter_type_validator(str, strict=False)
    def supercategory(self, v):
        self.__supercategory = v if v else "object"

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    @setter_type_validator(str, strict=False)
    def name(self, v):
        self.__name = v

    @property
    def keypoints(self) -> list[str]:
        return self.__keypoints

    @keypoints.setter
    @setter_type_validator(list)
    def keypoints(self, v):
        self.__keypoints = v

    @property
    def skeleton(self) -> list[list[int]]:
        return self.__skeleton

    @skeleton.setter
    @setter_type_validator(list)
    def skeleton(self, v):
        self.__skeleton = v

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
        if not isinstance(other, CategoryInfo):
            return False

        if self.task == TaskType.CLASSIFICATION:
            return self.name == other.name and self.supercategory == other.supercategory
        elif self.task == TaskType.OBJECT_DETECTION:
            return self.name == other.name and self.supercategory == other.supercategory
        elif self.task == TaskType.SEMANTIC_SEGMENTATION:
            return self.name == other.name and self.supercategory == other.supercategory
        elif self.task == TaskType.INSTANCE_SEGMENTATION:
            return self.name == other.name and self.supercategory == other.supercategory
        elif self.task == TaskType.TEXT_RECOGNITION:
            return self.name == other.name and self.supercategory == other.supercategory
        elif self.task == TaskType.KEYPOINT_DETECTION:
            return (
                self.name == other.name
                and self.supercategory == other.supercategory
                and self.keypoints == other.keypoints
                and self.skeleton == other.skeleton
            )
        else:
            raise NotImplementedError

    # factories
    @classmethod
    def new(
        cls,
        task: Union[str, TaskType],
        *args,
        **kwargs,
    ) -> "CategoryInfo":
        """Category Format

        Args:
            task (Union[str, TaskType]): task type.
            *args: arguments.
            **kwargs: keyword arguments.

        Returns:
            Category: category class
        """
        if task not in list(TaskType):
            raise ValueError(f"Invalid task type: {task}" f"Available task types: {list(TaskType)}")

        return getattr(cls, task.lower())(*args, **kwargs)

    @classmethod
    def classification(
        cls, category_id: int, name: str, supercategory: str = "object"
    ) -> "CategoryInfo":
        """Classification Category Format

        Args:
            category_id (int): category id. natural number.
            name (str): category name.
            supercategory (str, optional): supercategory name. Defaults to "object".

        Returns:
            Category: category class
        """
        return cls(
            task=TaskType.CLASSIFICATION,
            category_id=category_id,
            name=name,
            supercategory=supercategory,
        )

    @classmethod
    def object_detection(
        cls,
        category_id: int,
        name: str,
        supercategory: str = "object",
    ) -> "CategoryInfo":
        """Object Detection Category Format

        Args:
            category_id (int): category id. natural number.
            name (str): category name.
            supercategory (str, optional): supercategory name. Defaults to "object".

        Returns:
            Category: category class
        """
        return cls(
            task=TaskType.OBJECT_DETECTION,
            category_id=category_id,
            name=name,
            supercategory=supercategory,
        )

    @classmethod
    def semantic_segmentation(
        cls,
        category_id: int,
        name: str,
        supercategory: str = "object",
    ) -> "CategoryInfo":
        """Segmentation Category Format

        Args:
            category_id (int): category id. natural number.
            name (str): category name.
            supercategory (str, optional): supercategory name. Defaults to "object".

        Returns:
            Category: category class
        """
        return cls(
            task=TaskType.SEMANTIC_SEGMENTATION,
            category_id=category_id,
            name=name,
            supercategory=supercategory,
        )

    @classmethod
    def instance_segmentation(
        cls,
        category_id: int,
        name: str,
        supercategory: str = "object",
    ) -> "CategoryInfo":
        """Instance Category Format

        Args:
            category_id (int): category id. natural number.
            name (str): category name.
            supercategory (str, optional): supercategory name. Defaults to "object".

        Returns:
            Category: category class
        """
        return cls(
            task=TaskType.INSTANCE_SEGMENTATION,
            category_id=category_id,
            name=name,
            supercategory=supercategory,
        )

    @classmethod
    def keypoint_detection(
        cls,
        category_id: int,
        name: str,
        keypoints: list[str],
        skeleton: list[list[int]],
        supercategory: str = "object",
    ) -> "CategoryInfo":
        """Keypoint Detection Category Format

        Args:
            category_id (int): category id. natural number.
            name (str): category name.
            keypoints (list[str]): category name.
            skeleton (list[list[int]]): skeleton edges.
            supercategory (str, optional): supercategory name. Defaults to "object".

        Returns:
            Category: category class
        """
        return cls(
            task=TaskType.KEYPOINT_DETECTION,
            category_id=category_id,
            name=name,
            supercategory=supercategory,
            keypoints=keypoints,
            skeleton=skeleton,
        )

    @classmethod
    def text_recognition(
        cls,
        category_id: int,
        name: str,
        supercategory: str = "object",
    ) -> "CategoryInfo":
        """Text Recognition Category Format

        Args:
            category_id (int): category id. natural number.
            name (str): category name.
            supercategory (str, optional): supercategory name. Defaults to "object".

        Returns:
            Category: category class
        """
        return cls(
            task=TaskType.TEXT_RECOGNITION,
            category_id=category_id,
            name=name,
            supercategory=supercategory,
        )

    def to_dict(self) -> dict:
        """Get Dictionary of Category

        Returns:
            dict: annotation dictionary.
        """

        cat = {
            "category_id": self.category_id,
            "supercategory": self.supercategory,
            "name": self.name,
        }

        if self.keypoints is not None:
            cat["keypoints"] = self.keypoints
        if self.skeleton is not None:
            cat["skeleton"] = self.skeleton

        return cat
