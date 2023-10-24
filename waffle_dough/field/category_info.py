from typing import Optional

from waffle_dough.field.base_field import BaseField
from waffle_dough.type import TaskType


class CategoryInfo(BaseField):
    name: str
    supercategory: str = "object"
    keypoints: Optional[list[str]] = None
    skeleton: Optional[list[list[int]]] = None

    @classmethod
    def classification(cls, name: str, supercategory: str = "object") -> "CategoryInfo":
        """Classification Category Format

        Args:
            name (str): category name.
            supercategory (str, optional): supercategory name. Defaults to "object".

        Returns:
            Category: category class
        """
        return cls(
            task=TaskType.CLASSIFICATION,
            name=name,
            supercategory=supercategory,
        )

    @classmethod
    def object_detection(
        cls,
        name: str,
        supercategory: str = "object",
    ) -> "CategoryInfo":
        """Object Detection Category Format

        Args:
            name (str): category name.
            supercategory (str, optional): supercategory name. Defaults to "object".

        Returns:
            Category: category class
        """
        return cls(
            task=TaskType.OBJECT_DETECTION,
            name=name,
            supercategory=supercategory,
        )

    @classmethod
    def semantic_segmentation(
        cls,
        name: str,
        supercategory: str = "object",
    ) -> "CategoryInfo":
        """Segmentation Category Format

        Args:
            name (str): category name.
            supercategory (str, optional): supercategory name. Defaults to "object".

        Returns:
            Category: category class
        """
        return cls(
            task=TaskType.SEMANTIC_SEGMENTATION,
            name=name,
            supercategory=supercategory,
        )

    @classmethod
    def instance_segmentation(
        cls,
        name: str,
        supercategory: str = "object",
    ) -> "CategoryInfo":
        """Instance Category Format

        Args:
            name (str): category name.
            supercategory (str, optional): supercategory name. Defaults to "object".

        Returns:
            Category: category class
        """
        return cls(
            task=TaskType.INSTANCE_SEGMENTATION,
            name=name,
            supercategory=supercategory,
        )

    @classmethod
    def keypoint_detection(
        cls,
        name: str,
        keypoints: list[str],
        skeleton: list[list[int]],
        supercategory: str = "object",
    ) -> "CategoryInfo":
        """Keypoint Detection Category Format

        Args:
            name (str): category name.
            keypoints (list[str]): category name.
            skeleton (list[list[int]]): skeleton edges.
            supercategory (str, optional): supercategory name. Defaults to "object".

        Returns:
            Category: category class
        """
        return cls(
            task=TaskType.KEYPOINT_DETECTION,
            name=name,
            supercategory=supercategory,
            keypoints=keypoints,
            skeleton=skeleton,
        )

    @classmethod
    def text_recognition(
        cls,
        name: str,
        supercategory: str = "object",
    ) -> "CategoryInfo":
        """Text Recognition Category Format

        Args:
            name (str): category name.
            supercategory (str, optional): supercategory name. Defaults to "object".

        Returns:
            Category: category class
        """
        return cls(
            task=TaskType.TEXT_RECOGNITION,
            name=name,
            supercategory=supercategory,
        )

    @classmethod
    def regression(
        cls,
        name: str,
        supercategory: str = "object",
    ) -> "CategoryInfo":
        """Regression Category Format

        Args:
            name (str): category name.
            supercategory (str, optional): supercategory name. Defaults to "object".

        Returns:
            Category: category class
        """
        return cls(
            task=TaskType.REGRESSION,
            name=name,
            supercategory=supercategory,
        )
