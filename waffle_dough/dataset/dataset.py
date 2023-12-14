"""
Waffle Dough Dataset

This module contains the Waffle Dough Dataset class.


Example:
    >>> from waffle_dough.dataset import WaffleDataset
    >>> dataset = WaffleDataset.new(
    ...     name="my_dataset",
    ...     task="classification",
    ...     root_dir="~/datasets",
    ... )
    >>> dataset.add_category(
    ...     CategoryInfo.classification(
    ...         name="cat",
    ...         label=0,

    )

"""
import os
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Union

from waffle_utils.file import io
from waffle_utils.logger import datetime_now

from waffle_dough.database.service import DatabaseService
from waffle_dough.dataset.adapter import CocoAdapter
from waffle_dough.exception.dataset_exception import *
from waffle_dough.field import (
    AnnotationInfo,
    CategoryInfo,
    ImageInfo,
    UpdateAnnotationInfo,
    UpdateCategoryInfo,
    UpdateImageInfo,
)
from waffle_dough.type import SplitType, TaskType


@dataclass
class DatasetInfo:
    name: str
    task: str
    categories: list[dict]
    created_at: str
    updated_at: str

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(dataset_info: dict):
        return DatasetInfo(
            name=dataset_info["name"],
            task=dataset_info["task"],
            categories=dataset_info["categories"],
            created_at=dataset_info["created_at"],
            updated_at=dataset_info["updated_at"],
        )


class WaffleDataset:
    DATASET_INFO_FILE_NAME = "dataset.yaml"
    DATABASE_FILE_NAME = "database.sqlite3"
    IMAGE_DIR_NAME = "images"

    def __init__(
        self,
        name: str,
        task: Union[str, TaskType] = None,
        root_dir: Union[str, Path] = None,
    ):
        self.name = name
        self.root_dir = root_dir

        if self.initialized():
            dataset_info = self.get_dataset_info()
            if task is not None and task.lower() != dataset_info.task:
                raise DatasetTaskError(f"Invalid task: {task}")
            self.task = dataset_info.task
        else:
            if task is None:
                raise DatasetTaskError(f"Task is not specified")
            self.task = task
            self.initialize()

    def __repr__(self) -> str:
        return f"WaffleDataset(name={self.name}, task={self.task}, root_dir={self.root_dir})"

    def __str__(self) -> str:
        return f"WaffleDataset(name={self.name}, task={self.task}, root_dir={self.root_dir})"

    def initialize(self):
        io.make_directory(self.dataset_dir)
        io.make_directory(self.image_dir)
        self.create_dataset_info()

    def initialized(self) -> bool:
        return self.dataset_info_file_path.exists()

    def create_dataset_info(self) -> DatasetInfo:
        dataset_info = DatasetInfo(
            name=self.name,
            task=self.task,
            categories=[category.to_dict() for category in self.categories],
            created_at=datetime_now(),
            updated_at=datetime_now(),
        )
        io.save_yaml(dataset_info.to_dict(), self.dataset_info_file_path)
        return dataset_info

    def get_dataset_info(self) -> DatasetInfo:
        return DatasetInfo.from_dict(io.load_yaml(self.dataset_info_file_path))

    def update_dataset_info(self) -> DatasetInfo:
        dataset_info = self.get_dataset_info()
        dataset_info.categories = [category.to_dict() for category in self.categories]
        dataset_info.updated_at = datetime_now()
        io.save_yaml(dataset_info.to_dict(), self.dataset_info_file_path)
        return dataset_info

    def update_dataset_decorator(func):
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            self.update_dataset_info()
            return result

        return wrapper

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = str(name)

    @property
    def task(self) -> str:
        return self._task

    @task.setter
    def task(self, task: Union[str, TaskType]):
        if task not in list(TaskType):
            raise DatasetTaskError(f"Invalid task: {task}")
        self._task = task.lower()

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    @root_dir.setter
    def root_dir(self, root_dir: str):
        self._root_dir = self.parse_root_dir(root_dir)

    @property
    def dataset_dir(self) -> Path:
        return self.root_dir / self.name

    @property
    def dataset_info_file_path(self) -> Path:
        return self.dataset_dir / self.DATASET_INFO_FILE_NAME

    @property
    def image_dir(self) -> Path:
        return self.dataset_dir / self.IMAGE_DIR_NAME

    @property
    def database_file_path(self) -> Path:
        return self.dataset_dir / self.DATABASE_FILE_NAME

    @cached_property
    def database_service(self) -> DatabaseService:
        return DatabaseService(str(self.database_file_path), image_directory=self.image_dir)

    @property
    def category_dict(self) -> dict[str, CategoryInfo]:
        return self.database_service.get_categories()

    @property
    def categories(self) -> list[CategoryInfo]:
        return list(self.category_dict.values())

    @property
    def category_names(self) -> list[str]:
        return [category.name for category in self.categories]

    @property
    def image_dict(self) -> dict[str, ImageInfo]:
        return self.database_service.get_images()

    @property
    def images(self) -> list[ImageInfo]:
        return list(self.image_dict.values())

    @property
    def annotation_dict(self) -> dict[str, AnnotationInfo]:
        return self.database_service.get_annotations()

    @property
    def annotations(self) -> list[AnnotationInfo]:
        return list(self.database_service.get_annotations().values())

    # methods (CRUD)
    @update_dataset_decorator
    def add_category(self, category_info: Union[CategoryInfo, list[CategoryInfo]]):
        category_infos = category_info if isinstance(category_info, list) else [category_info]
        for category_info in category_infos:
            self.database_service.add_category(category_info)

    @update_dataset_decorator
    def add_image(self, image: Union[str, Path], image_info: Union[ImageInfo, list[ImageInfo]]):
        images = image if isinstance(image, list) else [image]
        image_infos = image_info if isinstance(image_info, list) else [image_info]
        for image, image_info in zip(images, image_infos):
            self.database_service.add_image(image, image_info)

    @update_dataset_decorator
    def add_annotation(self, annotation_info: Union[AnnotationInfo, list[AnnotationInfo]]):
        annotation_infos = (
            annotation_info if isinstance(annotation_info, list) else [annotation_info]
        )
        for annotation_info in annotation_infos:
            self.database_service.add_annotation(annotation_info)

    def get_image_dict(
        self,
        image_id: Union[str, list[str]] = None,
        category_id: Union[str, list[str]] = None,
        split_type: Union[str, SplitType] = None,
    ) -> dict[str, ImageInfo]:
        if category_id is None:
            images = self.database_service.get_images(image_id=image_id, split_type=split_type)
        else:
            images = self.database_service.get_images_by_category_id(
                category_id=category_id, split_type=split_type
            )
        return images

    def get_images(
        self,
        image_id: Union[str, list[str]] = None,
        category_id: Union[str, list[str]] = None,
        split_type: Union[str, SplitType] = None,
    ) -> list[ImageInfo]:
        return list(
            self.get_image_dict(
                image_id=image_id, category_id=category_id, split_type=split_type
            ).values()
        )

    def get_annotation_dict(
        self,
        image_id: Union[str, list[str]] = None,
        category_id: Union[str, list[str]] = None,
        split_type: Union[str, SplitType] = None,
    ) -> dict[str, AnnotationInfo]:
        images = self.get_images(image_id=image_id, category_id=category_id, split_type=split_type)
        annotations = self.database_service.get_annotations_by_image_id(
            image_id=[image.id for image in images]
        )
        return annotations

    def get_annotations(
        self,
        image_id: Union[str, list[str]] = None,
        category_id: Union[str, list[str]] = None,
        split_type: Union[str, SplitType] = None,
    ) -> list[AnnotationInfo]:
        return list(
            self.get_annotation_dict(
                image_id=image_id, category_id=category_id, split_type=split_type
            ).values()
        )

    def get_category_dict(
        self, category_id: Union[str, list[str]] = None
    ) -> dict[str, CategoryInfo]:
        return self.database_service.get_categories(category_id=category_id)

    def get_categories(self, category_id: Union[str, list[str]] = None) -> list[CategoryInfo]:
        return list(self.get_category_dict(category_id=category_id).values())

    @update_dataset_decorator
    def update_image(self, image_id: str, update_image_info: UpdateImageInfo) -> ImageInfo:
        return self.database_service.update_image(image_id, update_image_info)

    @update_dataset_decorator
    def update_category(
        self, category_id: str, update_category_info: UpdateCategoryInfo
    ) -> CategoryInfo:
        return self.database_service.update_category(category_id, update_category_info)

    @update_dataset_decorator
    def update_annotation(
        self, annotation_id: str, update_annotation_info: UpdateAnnotationInfo
    ) -> AnnotationInfo:
        return self.database_service.update_annotation(annotation_id, update_annotation_info)

    @update_dataset_decorator
    def delete_image(self, image_id: str):
        self.database_service.delete_image(image_id)

    @update_dataset_decorator
    def delete_category(self, category_id: str):
        self.database_service.delete_category(category_id)

    @update_dataset_decorator
    def delete_annotation(self, annotation_id: str):
        self.database_service.delete_annotation(annotation_id)

    # methods (class methods)
    @classmethod
    def get_dataset_list(cls, root_dir: str = None) -> list[str]:
        root_dir = cls.parse_root_dir(root_dir)

        dataset_list = []
        if not root_dir.exists():
            return dataset_list

        for dataset_dir in Path(root_dir).iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset_info_file_path = dataset_dir / cls.DATASET_INFO_FILE_NAME
            if dataset_info_file_path.exists():
                dataset_list.append(Path(dataset_dir).name)
        return dataset_list

    @classmethod
    def parse_root_dir(cls, root_dir: str = None) -> Path:
        if root_dir is None:
            root_dir = os.environ.get("WAFFLE_DATASET_ROOT_DIR", "datasets")
        return Path(root_dir).absolute()

    @classmethod
    def new(
        cls,
        name: str,
        task: Union[str, TaskType],
        root_dir: Union[str, Path] = None,
    ) -> "WaffleDataset":
        if name in WaffleDataset.get_dataset_list(root_dir=root_dir):
            raise DatasetAlreadyExistsError(f"Dataset '{name}' already exists")
        dataset = WaffleDataset(name, task, root_dir=root_dir)
        return dataset

    @classmethod
    def load(
        cls,
        name: str,
        root_dir: Union[str, Path] = None,
    ) -> "WaffleDataset":
        if name not in WaffleDataset.get_dataset_list(root_dir=root_dir):
            raise DatasetNotFoundError(f"Dataset '{name}' does not exists")
        dataset = WaffleDataset(name, root_dir=root_dir)
        return dataset

    @classmethod
    def delete(
        cls,
        name: str,
        root_dir: Union[str, Path] = None,
    ):
        dataset = WaffleDataset.load(name, root_dir=root_dir)
        io.remove_directory(dataset.dataset_dir, recursive=True)

    @classmethod
    def copy(
        cls,
        src_name: str,
        dst_name: str,
        root_dir: Union[str, Path] = None,
    ) -> "WaffleDataset":
        src_dataset = WaffleDataset.load(src_name, root_dir=root_dir)
        dst_dataset = WaffleDataset.new(dst_name, src_dataset.task, root_dir=root_dir)
        io.copy_files_to_directory(src_dataset.dataset_dir, dst_dataset.dataset_dir)
        return dst_dataset

    @classmethod
    def from_coco(
        cls,
        name: str,
        task: Union[str, TaskType],
        coco_file_path: Union[str, Path],
        coco_image_dir: Union[str, Path],
        root_dir: Union[str, Path] = None,
    ) -> "WaffleDataset":
        dataset = WaffleDataset.new(name, task, root_dir=root_dir)

        try:
            adapter = CocoAdapter.from_target(coco_file_path, task=task)

            dataset.add_category(list(adapter.categories.values()))
            dataset.add_image(
                list(
                    map(
                        lambda image: Path(coco_image_dir, image.original_file_name),
                        adapter.images.values(),
                    )
                ),
                list(adapter.images.values()),
            )
            dataset.add_annotation(list(adapter.annotations.values()))

            return dataset

        except Exception as e:
            WaffleDataset.delete(name, root_dir=root_dir)
            raise e
