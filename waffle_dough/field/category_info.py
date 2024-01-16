from typing import ClassVar, Optional

from pydantic import BaseModel, Field, field_validator

from waffle_dough.field.base_field import BaseField
from waffle_dough.field.validator.category_validator import (
    validate_keypoints,
    validate_skeleton,
)
from waffle_dough.type import TaskType


class CategoryInfo(BaseField):
    name: str = Field(..., kw_only=True)
    supercategory: str = Field(..., kw_only=True)
    keypoints: Optional[list[str]] = Field(None, kw_only=True)
    skeleton: Optional[list[list[int]]] = Field(None, kw_only=True)

    extra_required_fields: ClassVar[dict[TaskType, list[str]]] = {
        TaskType.CLASSIFICATION: ["name"],
        TaskType.OBJECT_DETECTION: ["name"],
        TaskType.SEMANTIC_SEGMENTATION: ["name"],
        TaskType.INSTANCE_SEGMENTATION: ["name"],
        TaskType.KEYPOINT_DETECTION: ["name", "keypoints", "skeleton"],
        TaskType.TEXT_RECOGNITION: ["name"],
        TaskType.REGRESSION: ["name"],
    }

    @field_validator("keypoints")
    def check_keypoints(cls, v):
        return validate_keypoints(v)

    @field_validator("skeleton")
    def check_skeleton(cls, v, values):
        return validate_skeleton(v, values.data.get("keypoints", None))

    @classmethod
    def classification(cls, *, name: str, supercategory: str = "object") -> "CategoryInfo":
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
        *,
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


class UpdateCategoryInfo(BaseModel):
    name: Optional[str] = Field(None)
    supercategory: Optional[str] = Field(None)
    keypoints: Optional[list[str]] = Field(None)
    skeleton: Optional[list[list[int]]] = Field(None)

    @field_validator("keypoints")
    def check_keypoints(cls, v):
        return validate_keypoints(v)

    @field_validator("skeleton")
    def check_skeleton(cls, v, values):
        return validate_skeleton(v, values.data.get("keypoints", None))

    @classmethod
    def classification(
        cls, name: Optional[str] = None, supercategory: Optional[str] = None
    ) -> "UpdateCategoryInfo":
        return cls(
            name=name,
            supercategory=supercategory,
        )

    @classmethod
    def object_detection(
        cls, name: Optional[str] = None, supercategory: Optional[str] = None
    ) -> "UpdateCategoryInfo":
        return cls(
            name=name,
            supercategory=supercategory,
        )

    @classmethod
    def semantic_segmentation(
        cls, name: Optional[str] = None, supercategory: Optional[str] = None
    ) -> "UpdateCategoryInfo":
        return cls(
            name=name,
            supercategory=supercategory,
        )

    @classmethod
    def instance_segmentation(
        cls, name: Optional[str] = None, supercategory: Optional[str] = None
    ) -> "UpdateCategoryInfo":
        return cls(
            name=name,
            supercategory=supercategory,
        )

    @classmethod
    def keypoint_detection(
        cls,
        name: Optional[str] = None,
        keypoints: Optional[list[str]] = None,
        skeleton: Optional[list[list[int]]] = None,
        supercategory: Optional[str] = None,
    ) -> "UpdateCategoryInfo":
        return cls(
            name=name,
            supercategory=supercategory,
            keypoints=keypoints,
            skeleton=skeleton,
        )

    @classmethod
    def text_recognition(
        cls, name: Optional[str] = None, supercategory: Optional[str] = None
    ) -> "UpdateCategoryInfo":
        return cls(
            name=name,
            supercategory=supercategory,
        )

    @classmethod
    def regression(
        cls, name: Optional[str] = None, supercategory: Optional[str] = None
    ) -> "UpdateCategoryInfo":
        return cls(
            name=name,
            supercategory=supercategory,
        )
