from abc import abstractmethod
from typing import Union
from uuid import uuid4

from pydantic import BaseModel
from waffle_utils.file import io

from waffle_dough.type import TaskType


class BaseField(BaseModel):
    id: str = None
    task: str

    def __init__(self, task: Union[str, TaskType], **data):
        if task not in list(TaskType):
            raise ValueError(f"Invalid task type: {task}" f"Available task types: {list(TaskType)}")
        task = task.upper()

        super().__init__(**data, task=task)

        if self.id is None:
            self.id = str(uuid4())

    def __eq__(self, __value: object) -> bool:
        d1 = self.to_dict()

        if isinstance(__value, dict):
            d2 = __value
        elif isinstance(__value, self.__class__):
            d2 = __value.to_dict()
        else:
            return False

        d1.pop("id")
        d2.pop("id")

        return d1 == d2

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        d.update(kwargs)
        return cls(**d)
