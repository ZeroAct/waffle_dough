from abc import abstractmethod
from typing import ClassVar, Union
from uuid import uuid4

import pydantic_core
from pydantic import BaseModel, Field, field_validator
from waffle_utils.file import io

from waffle_dough.exception import (
    FieldException,
    FieldMissingError,
    FieldTaskError,
    FieldValidationError,
)
from waffle_dough.type import TaskType


class BaseField(BaseModel):
    id: str = Field(..., default_factory=lambda: str(uuid4()), kw_only=True)
    task: str = Field(..., kw_only=True)

    extra_required_fields: ClassVar[dict[TaskType, list[str]]] = {}

    def __init__(self, **kwargs):
        try:
            super().__init__(**kwargs)
        except pydantic_core.ValidationError as e:
            task = kwargs.get("task", None)
            if task is None:
                raise FieldMissingError("task must be given")

            missing_keys = [
                key
                for key, f in self.model_fields.items()
                if f.is_required() and kwargs.get(key, None) is None
            ]
            if task in self.extra_required_fields:
                missing_keys += [
                    key for key in self.extra_required_fields[task] if kwargs.get(key, None) is None
                ]
            missing_keys = list(set(missing_keys))

            msg = "\n"
            if len(missing_keys) > 0:
                msg += f"Missing required fields: {missing_keys}\n"

                msg += f"Given fields:\n"
                for key, f in self.model_fields.items():
                    msg += f" - {key}[{f.annotation}]: {kwargs.get(key, None)}\n"

                raise FieldMissingError(msg)
            else:
                raise FieldValidationError(e)
        except Exception as e:
            raise e

    @field_validator("task")
    def _check_task_before(cls, v, values):
        if v not in list(TaskType):
            raise FieldTaskError(f"task must be one of {list(TaskType)}: {v}")
        return v

    def __new__(cls, *args, **kwargs):
        if cls is BaseField:
            raise FieldException("BaseField cannot be instantiated")
        return super().__new__(cls)

    def __eq__(self, __value: object) -> bool:
        d1 = self.to_dict()

        if isinstance(__value, dict):
            d2 = __value.copy()
        elif isinstance(__value, self.__class__):
            d2 = __value.to_dict()
        else:
            return False

        d1.pop("id")
        d2.pop("id")

        return d1 == d2

    def __ne__(self, __value: object) -> bool:
        return not self.__eq__(__value)

    def to_dict(self) -> dict:
        """Convert to dict.

        Returns:
            dict: dict.
        """

        d = self.model_dump()
        for key in list(d.keys()):
            if d.get(key, None) is None:
                d.pop(key)
        return d

    @classmethod
    def from_dict(cls, task: Union[str, TaskType], d: dict) -> "BaseField":
        """Create a field from a dict.

        Args:
            task (Union[str, TaskType]): task type.
            d (dict): dict.

        Returns:
            BaseField: field.
        """
        d = d.copy()

        if hasattr(d, "task") and task != d["task"]:
            raise FieldTaskError(
                f"Given task type is not matched with the task type of the given dict: {task} != {d['task']}"
            )
        else:
            d["task"] = task

        task = d.pop("task")
        return getattr(cls, task.lower())(**d)
