from typing import Union

from waffle_utils.hook import BaseHook

from waffle_dough.field import AnnotationInfo, CategoryInfo, ImageInfo
from waffle_dough.type import TaskType

from .callback import DatasetAdapterProgressCallback


class BaseAdapter(BaseHook):
    def __init__(
        self,
        image_dict: dict[str, ImageInfo] = None,
        annotation_dict: dict[str, AnnotationInfo] = None,
        category_dict: dict[str, CategoryInfo] = None,
        task: Union[str, TaskType] = TaskType.OBJECT_DETECTION,
        callbacks: list[callable] = None,
    ):
        callbacks = [
            DatasetAdapterProgressCallback(),
        ] + (callbacks if callbacks else [])

        super().__init__(callbacks=callbacks)

        self.image_dict: dict[str, ImageInfo] = image_dict or {}
        self.annotation_dict: dict[str, AnnotationInfo] = annotation_dict or {}
        self.category_dict: dict[str, CategoryInfo] = category_dict or {}
        self.task: Union[str, TaskType] = TaskType.from_str(task)

    def on_loop_start(self, total_steps: int):
        pass

    def on_loop_end(self):
        pass

    def on_step_start(self):
        pass

    def on_step_end(self, current_step: int):
        pass
