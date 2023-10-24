from typing import Optional

from pydantic import validator
from waffle_utils.logger import datetime_now

from waffle_dough.field.base_field import BaseField
from waffle_dough.type import TaskType


class ImageInfo(BaseField):
    file_name: str
    width: int
    height: int
    original_file_name: Optional[str] = None
    date_captured: Optional[str] = None

    task: Optional[str] = None

    def __init__(self, **data):
        if not hasattr(data, "task"):
            data.update({"task": TaskType.AGNOSTIC.upper()})
        super().__init__(**data)
