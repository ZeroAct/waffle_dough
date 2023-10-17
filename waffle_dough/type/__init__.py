"""
Set of types for waffle_dough
"""
from .data_type import DataType
from .task_type import TaskType


def get_data_types():
    return list(map(lambda x: x.value, list(DataType)))


def get_task_types():
    return list(map(lambda x: x.value, list(TaskType)))


__all__ = [
    "DataType",
    "TaskType",
    "get_data_types",
    "get_task_types"
]
