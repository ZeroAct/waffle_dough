from typing import Optional, Union

from pydantic import Field, field_validator

from waffle_dough.field.base_field import BaseField
from waffle_dough.type import SplitType, TaskType, get_split_types


class ImageInfo(BaseField):
    file_name: str = Field(...)
    width: int = Field(...)
    height: int = Field(...)
    original_file_name: Optional[str] = Field(None)
    date_captured: Optional[str] = Field(None)
    labeled: bool = Field(False)
    split: SplitType = Field(SplitType.UNSET.lower())

    def __init__(
        self,
        file_name: str,
        width: int,
        height: int,
        task: Union[str, TaskType] = TaskType.AGNOSTIC,
        original_file_name: Optional[str] = None,
        date_captured: Optional[str] = None,
        *args,
        **kwargs,
    ) -> "ImageInfo":
        """Image Information Field

        Args:
            task (Union[str, TaskType]): task type.
            file_name (str): file name.
            width (int): width.
            height (int): height.
            original_file_name (Optional[str], optional): original file name. Defaults to None.
            date_captured (Optional[str], optional): date captured. Defaults to None.

        Returns:
            ImageInfo: image information field.
        """
        super().__init__(
            task=task,
            file_name=file_name,
            width=width,
            height=height,
            original_file_name=original_file_name,
            date_captured=date_captured,
            *args,
            **kwargs,
        )

    @field_validator("width")
    def _check_width_after(cls, v):
        if v <= 0:
            raise ValueError(f"width must be positive: {v}")
        return v

    @field_validator("height")
    def _check_height_after(cls, v):
        if v <= 0:
            raise ValueError(f"height must be positive: {v}")
        return v

    @classmethod
    def agnostic(
        cls,
        *,
        file_name: str,
        width: int,
        height: int,
        original_file_name: Optional[str] = None,
        date_captured: Optional[str] = None,
    ) -> "ImageInfo":
        """Agnostic Image Format

        Args:
            file_name (str): file name.
            width (int): width.
            height (int): height.
            original_file_name (Optional[str], optional): original file name. Defaults to None.
            date_captured (Optional[str], optional): date captured. Defaults to None.

        Returns:
            ImageInfo: image information field.
        """
        return cls(
            task=TaskType.AGNOSTIC,
            file_name=file_name,
            width=width,
            height=height,
            original_file_name=original_file_name,
            date_captured=date_captured,
        )
