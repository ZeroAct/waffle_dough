from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, Field, field_validator

from waffle_dough.field.base_field import BaseField
from waffle_dough.field.validator.image_validator import (
    validate_height,
    validate_split,
    validate_width,
)
from waffle_dough.type import SplitType, TaskType, get_split_types


class ImageInfo(BaseField):
    ext: str = Field(...)
    width: int = Field(...)
    height: int = Field(...)
    original_file_name: str = Field(...)
    date_captured: Optional[str] = Field(None)
    split: SplitType = Field(SplitType.UNSET.lower())

    def __init__(
        self,
        ext: str,
        width: int,
        height: int,
        original_file_name: str,
        task: Union[str, TaskType] = TaskType.AGNOSTIC,
        date_captured: Optional[str] = None,
        *args,
        **kwargs,
    ) -> "ImageInfo":
        """Image Information Field

        Args:
            task (Union[str, TaskType]): task type.
            ext (str): file ext(including .).
            width (int): width.
            height (int): height.
            original_file_name (str): original file name.
            task (Union[str, TaskType], optional): task type. Defaults to TaskType.AGNOSTIC.
            date_captured (Optional[str], optional): date captured. Defaults to None.

        Returns:
            ImageInfo: image information field.
        """
        super().__init__(
            task=task,
            ext=ext,
            width=width,
            height=height,
            original_file_name=original_file_name,
            date_captured=date_captured,
            *args,
            **kwargs,
        )

    @field_validator("width")
    def _check_width(cls, v):
        return validate_width(v)

    @field_validator("height")
    def _check_height_after(cls, v):
        return validate_height(v)

    @field_validator("split")
    def _check_split(cls, v):
        return validate_split(v)

    @classmethod
    def agnostic(
        cls,
        *,
        width: int,
        height: int,
        original_file_name: str,
        date_captured: Optional[str] = None,
        split: SplitType = SplitType.UNSET,
    ) -> "ImageInfo":
        """Agnostic Image Format

        Args:
            width (int): width.
            height (int): height.
            original_file_name (str): original file name
            date_captured (Optional[str], optional): date captured. Defaults to None.

        Returns:
            ImageInfo: image information field.
        """
        return cls(
            task=TaskType.AGNOSTIC,
            ext=Path(original_file_name).suffix.lower(),
            width=width,
            height=height,
            original_file_name=original_file_name,
            date_captured=date_captured,
            split=split,
        )


class UpdateImageInfo(BaseModel):
    split: Optional[SplitType] = Field(None)

    @field_validator("split")
    def _check_split(cls, v):
        return validate_split(v)

    @classmethod
    def agnostic(cls, *, split: SplitType = None) -> "UpdateImageInfo":
        """Agnostic Image Format

        Args:
            split (SplitType, optional): split. Defaults to None.

        Returns:
            UpdateImageInfo: update image information field.
        """
        return cls(
            split=split,
        )
