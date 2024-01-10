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
import logging
import os
import random
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Union

from tqdm import tqdm
from waffle_utils.file import io
from waffle_utils.logger import datetime_now, initialize_logger

from waffle_dough.database.service import DatabaseService
from waffle_dough.dataset.adapter import CocoAdapter
from waffle_dough.dataset.util.iterator import Iterator
from waffle_dough.dataset.util.visualize import visualize
from waffle_dough.exception.base_exception import BaseException
from waffle_dough.exception.dataset_exception import *
from waffle_dough.field import (
    AnnotationInfo,
    CategoryInfo,
    ImageInfo,
    UpdateAnnotationInfo,
    UpdateCategoryInfo,
    UpdateImageInfo,
)
from waffle_dough.image import io as image_io
from waffle_dough.type import SplitType, TaskType

initialize_logger(
    console_level=os.environ.get("WAFFLE_DATASET_CONSOLE_LOG_LEVEL", "INFO"),
    file_level=os.environ.get("WAFFLE_DATASET_FILE_LOG_LEVEL", "INFO"),
    root_level=os.environ.get("WAFFLE_DATASET_ROOT_LOG_LEVEL", "INFO"),
)
logger = logging.getLogger(__name__)


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
        return self.dataset_info_file.exists()

    def create_dataset_info(self) -> DatasetInfo:
        dataset_info = DatasetInfo(
            name=self.name,
            task=self.task,
            categories=[category.to_dict() for category in self.categories],
            created_at=datetime_now(),
            updated_at=datetime_now(),
        )
        io.save_yaml(dataset_info.to_dict(), self.dataset_info_file)
        return dataset_info

    def get_dataset_info(self) -> DatasetInfo:
        return DatasetInfo.from_dict(io.load_yaml(self.dataset_info_file))

    def update_dataset_info(self) -> DatasetInfo:
        dataset_info = self.get_dataset_info()
        dataset_info.categories = [category.to_dict() for category in self.categories]
        dataset_info.updated_at = datetime_now()
        io.save_yaml(dataset_info.to_dict(), self.dataset_info_file)
        return dataset_info

    # decorators
    def update_dataset_decorator(func):
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            self.update_dataset_info()
            return result

        return wrapper

    def exception_decorator(func):
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except BaseException as e:
                raise e
            except Exception as e:
                exc = BaseException(e)
                exc.__traceback__ = e.__traceback__
                raise exc

        return wrapper

    # properties
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
    def dataset_info_file(self) -> Path:
        return self.dataset_dir / self.DATASET_INFO_FILE_NAME

    @property
    def image_dir(self) -> Path:
        return self.dataset_dir / self.IMAGE_DIR_NAME

    @property
    def database_file(self) -> Path:
        return self.dataset_dir / self.DATABASE_FILE_NAME

    @property
    def log_file(self) -> Path:
        return self.dataset_dir / "logs" / "dataset.log"

    @cached_property
    def database_service(self) -> DatabaseService:
        return DatabaseService(str(self.database_file), image_directory=self.image_dir)

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
    @exception_decorator
    def add_category(
        self, category_info: Union[CategoryInfo, list[CategoryInfo], dict, list[dict]]
    ) -> list[CategoryInfo]:
        category_infos = []
        for c in category_info if isinstance(category_info, list) else [category_info]:
            if isinstance(c, dict):
                c = CategoryInfo.from_dict(task=self.task, d=c)
            if c.task != self.task:
                raise DatasetTaskError(f"Invalid task: {c.task}")
            category_infos.append(c)

        for category_info in category_infos:
            self.database_service.add_category(category_info)

        return category_infos

    @update_dataset_decorator
    @exception_decorator
    def add_image(
        self,
        image: Union[str, Path],
        image_info: Union[ImageInfo, list[ImageInfo], dict, list[dict]] = None,
    ) -> list[ImageInfo]:
        images = image if isinstance(image, list) else [image]
        if image_info is None:
            image_infos = []
            for image in images:
                img = image_io.cv2_imread(image)
                image_infos.append(
                    ImageInfo(
                        ext=Path(image).suffix,
                        width=img.shape[1],
                        height=img.shape[0],
                        original_file_name=str(image),
                        task=TaskType.AGNOSTIC,
                    )
                )
        else:
            image_infos = []
            for image_info in image_info if isinstance(image_info, list) else [image_info]:
                if isinstance(image_info, dict):
                    image_info = ImageInfo.from_dict(task=TaskType.AGNOSTIC, d=image_info)
                if image_info.task != TaskType.AGNOSTIC:
                    raise DatasetTaskError(f"Invalid task: {image_info.task}")
                image_infos.append(image_info)

            if len(images) != len(image_infos):
                raise DatasetImportError(f"Number of images and image_infos must be same")

        for image, image_info in zip(images, image_infos):
            if isinstance(image_info, dict):
                image_info = ImageInfo.from_dict(task=TaskType.AGNOSTIC, d=image_info)
            self.database_service.add_image(image, image_info)

        return image_infos

    @update_dataset_decorator
    @exception_decorator
    def add_annotation(
        self, annotation_info: Union[AnnotationInfo, list[AnnotationInfo], dict, list[dict]]
    ) -> list[AnnotationInfo]:
        annotation_infos = []
        for a in annotation_info if isinstance(annotation_info, list) else [annotation_info]:
            if isinstance(a, dict):
                a = AnnotationInfo.from_dict(task=self.task, d=a)
            if a.task != self.task:
                raise DatasetTaskError(f"Invalid task: {a.task}")
            annotation_infos.append(a)

        for annotation_info in annotation_infos:
            self.database_service.add_annotation(annotation_info)

        return annotation_infos

    @exception_decorator
    def get_image_dict(
        self,
        image_id: Union[str, list[str]] = None,
        category_id: Union[str, list[str]] = None,
        split: Union[str, SplitType] = None,
    ) -> dict[str, ImageInfo]:
        if category_id is None:
            images = self.database_service.get_images(image_id=image_id, split=split)
        else:
            images = self.database_service.get_images_by_category_id(
                category_id=category_id, split=split
            )
        return images

    @exception_decorator
    def get_images(
        self,
        image_id: Union[str, list[str]] = None,
        category_id: Union[str, list[str]] = None,
        split: Union[str, SplitType] = None,
    ) -> list[ImageInfo]:
        return list(
            self.get_image_dict(image_id=image_id, category_id=category_id, split=split).values()
        )

    @exception_decorator
    def get_image_path(self, image: ImageInfo):
        return self.database_service.get_image_path(image)

    @exception_decorator
    def get_annotation_dict(
        self,
        image_id: Union[str, list[str]] = None,
        category_id: Union[str, list[str]] = None,
        split: Union[str, SplitType] = None,
    ) -> dict[str, AnnotationInfo]:
        images = self.get_images(image_id=image_id, category_id=category_id, split=split)
        annotations = self.database_service.get_annotations_by_image_id(
            image_id=[image.id for image in images]
        )
        return annotations

    @exception_decorator
    def get_annotations(
        self,
        image_id: Union[str, list[str]] = None,
        category_id: Union[str, list[str]] = None,
        split: Union[str, SplitType] = None,
    ) -> list[AnnotationInfo]:
        return list(
            self.get_annotation_dict(
                image_id=image_id, category_id=category_id, split=split
            ).values()
        )

    @exception_decorator
    def get_category_dict(
        self, category_id: Union[str, list[str]] = None
    ) -> dict[str, CategoryInfo]:
        return self.database_service.get_categories(category_id=category_id)

    @exception_decorator
    def get_categories(self, category_id: Union[str, list[str]] = None) -> list[CategoryInfo]:
        return list(self.get_category_dict(category_id=category_id).values())

    @exception_decorator
    def get_mapper(
        self,
        image_id: Union[str, list[str]] = None,
        category_id: Union[str, list[str]] = None,
        split: Union[str, SplitType] = None,
        labeled_only: bool = False,
    ) -> dict[str, dict]:
        image_dict = self.get_image_dict(image_id=image_id, category_id=category_id, split=split)
        category_dict = self.get_category_dict(category_id=category_id)

        mapper = {}
        for image_id, image in image_dict.items():
            annotations = self.get_annotations(image_id=image_id)
            categories = [category_dict[annotation.category_id] for annotation in annotations]
            if labeled_only and len(categories) == 0:
                continue
            mapper[image_id] = {
                "image_path": self.get_image_path(image),
                "image_info": image,
                "annotations": annotations,
                "categories": categories,
            }

        return mapper

    @update_dataset_decorator
    @exception_decorator
    def update_image(self, image_id: str, update_image_info: UpdateImageInfo) -> ImageInfo:
        return self.database_service.update_image(image_id, update_image_info)

    @update_dataset_decorator
    @exception_decorator
    def update_category(
        self, category_id: str, update_category_info: UpdateCategoryInfo
    ) -> CategoryInfo:
        return self.database_service.update_category(category_id, update_category_info)

    @update_dataset_decorator
    @exception_decorator
    def update_annotation(
        self, annotation_id: str, update_annotation_info: UpdateAnnotationInfo
    ) -> AnnotationInfo:
        return self.database_service.update_annotation(annotation_id, update_annotation_info)

    @update_dataset_decorator
    @exception_decorator
    def delete_image(self, image_id: str):
        self.database_service.delete_image(image_id)

    @update_dataset_decorator
    @exception_decorator
    def delete_category(self, category_id: str):
        self.database_service.delete_category(category_id)

    @update_dataset_decorator
    @exception_decorator
    def delete_annotation(self, annotation_id: str):
        self.database_service.delete_annotation(annotation_id)

    # methods (class methods)
    @classmethod
    @exception_decorator
    def get_dataset_list(cls, task: Union[str, TaskType] = None, root_dir: str = None) -> list[str]:
        root_dir = cls.parse_root_dir(root_dir)

        dataset_list = []
        if not root_dir.exists():
            return dataset_list

        for dataset_dir in Path(root_dir).iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset_info_file = dataset_dir / cls.DATASET_INFO_FILE_NAME
            if dataset_info_file.exists():
                dataset_info = DatasetInfo.from_dict(io.load_yaml(dataset_info_file))
                if task is None or TaskType.from_str(task) == dataset_info.task:
                    dataset_list.append(Path(dataset_dir).name)
        return dataset_list

    @classmethod
    @exception_decorator
    def parse_root_dir(cls, root_dir: str = None) -> Path:
        if root_dir is None:
            root_dir = os.environ.get("WAFFLE_DATASET_ROOT_DIR", "datasets")
        return Path(root_dir).absolute()

    @classmethod
    @exception_decorator
    def new(
        cls,
        name: str,
        task: Union[str, TaskType],
        root_dir: Union[str, Path] = None,
    ) -> "WaffleDataset":
        if name in WaffleDataset.get_dataset_list(root_dir=root_dir):
            raise DatasetAlreadyExistsError(f"Dataset '{name}' already exists")
        dataset = WaffleDataset(name, task, root_dir=root_dir)
        logger.info(f"Dataset created [{name}]\n{dataset}")
        return dataset

    @classmethod
    @exception_decorator
    def load(
        cls,
        name: str,
        root_dir: Union[str, Path] = None,
    ) -> "WaffleDataset":
        if name not in WaffleDataset.get_dataset_list(root_dir=root_dir):
            raise DatasetNotFoundError(f"Dataset '{name}' does not exists")
        dataset = WaffleDataset(name, root_dir=root_dir)
        logger.info(f"Dataset loaded [{name}]\n{dataset}")
        return dataset

    @classmethod
    @exception_decorator
    def delete(
        cls,
        name: str,
        root_dir: Union[str, Path] = None,
    ):
        dataset = WaffleDataset.load(name, root_dir=root_dir)
        io.remove_directory(dataset.dataset_dir, recursive=True)
        logger.info(f"Dataset deleted [{name}]\n{dataset}")

    @classmethod
    @exception_decorator
    def copy(
        cls,
        src_name: str,
        dst_name: str,
        root_dir: Union[str, Path] = None,
    ) -> "WaffleDataset":
        src_dataset = WaffleDataset.load(src_name, root_dir=root_dir)
        dst_dataset = WaffleDataset.new(dst_name, src_dataset.task, root_dir=root_dir)

        try:
            io.copy_files_to_directory(src_dataset.image_dir, dst_dataset.image_dir)
        except FileNotFoundError:
            pass
        io.copy_file(src_dataset.database_file, dst_dataset.database_file)
        dst_dataset.update_dataset_info()
        logger.info(f"Dataset copied [{src_name} -> {dst_name}]\n{dst_dataset}")

        return dst_dataset

    @classmethod
    def from_coco(
        cls,
        name: str,
        task: Union[str, TaskType],
        coco: Union[str, Path, dict],
        coco_image_dir: Union[str, Path],
        root_dir: Union[str, Path] = None,
    ) -> "WaffleDataset":
        dataset = WaffleDataset.new(name, task, root_dir=root_dir)

        try:
            adapter = CocoAdapter.from_target(coco, task=task)

            dataset.add_category(list(adapter.categories.values()))

            image_path = []
            image_infos = []
            for image in adapter.images.values():
                image_path.append(Path(coco_image_dir, image.original_file_name))
                image_infos.append(image)
            dataset.add_image(image_path, image_infos)
            dataset.add_annotation(list(adapter.annotations.values()))

            return dataset

        except Exception as e:
            WaffleDataset.delete(name, root_dir=root_dir)
            raise e

    @exception_decorator
    def random_split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ):
        image_ids = list(self.get_mapper(labeled_only=True).keys())
        if len(image_ids) == 0:
            raise DatasetEmptyError(f"There are no labeled images")

        random.seed(seed)
        random.shuffle(image_ids)

        # make sure that the ratios are valid
        train_ratio = float(train_ratio) if train_ratio else 0.0
        val_ratio = float(val_ratio) if val_ratio else 0.0
        test_ratio = float(test_ratio) if test_ratio else 0.0
        if (
            any([ratio < 0 for ratio in [train_ratio, val_ratio, test_ratio]])
            or sum([train_ratio, val_ratio, test_ratio]) == 0
        ):
            raise DatasetSplitError(
                "Ratio must be non-negative float or int, and their sum must be positive"
            )

        # initialize split
        for image_id in (
            list(self.get_image_dict(split=SplitType.TRAIN).keys())
            + list(self.get_image_dict(split=SplitType.VALIDATION).keys())
            + list(self.get_image_dict(split=SplitType.TEST).keys())
        ):
            self.update_image(image_id, UpdateImageInfo(split=SplitType.UNSET))

        # make the sum of ratios to 1
        ratio_sum = train_ratio + val_ratio + test_ratio
        train_ratio = train_ratio / ratio_sum
        val_ratio = val_ratio / ratio_sum
        test_ratio = test_ratio / ratio_sum

        train_image_ids = image_ids[: int(len(image_ids) * train_ratio)]
        val_image_ids = image_ids[
            int(len(image_ids) * train_ratio) : int(len(image_ids) * (train_ratio + val_ratio))
        ]
        test_image_ids = image_ids[int(len(image_ids) * (train_ratio + val_ratio)) :]

        for image_id in train_image_ids:
            self.update_image(image_id, UpdateImageInfo(split=SplitType.TRAIN))
        for image_id in val_image_ids:
            self.update_image(image_id, UpdateImageInfo(split=SplitType.VALIDATION))
        for image_id in test_image_ids:
            self.update_image(image_id, UpdateImageInfo(split=SplitType.TEST))

        logger.info(
            f"Dataset splitted [{self.name}]\ntrain: {len(train_image_ids)}, val: {len(val_image_ids)}, test: {len(test_image_ids)}"
        )

    @exception_decorator
    def get_dataset_iterator(
        self,
        image_id: Union[str, list[str]] = None,
        category_id: Union[str, list[str]] = None,
        split: Union[str, SplitType] = None,
        preprocess=None,
        augment=None,
    ) -> Iterator:
        return Iterator(
            self.get_mapper(image_id=image_id, category_id=category_id, split=split),
            preprocess=preprocess,
            augment=augment,
        )

    @exception_decorator
    def visualize(
        self,
        image_id: Union[str, list[str]] = None,
        category_id: Union[str, list[str]] = None,
        split: Union[str, SplitType] = None,
        result_dir: Union[str, Path] = None,
        show: bool = False,
    ) -> Path:
        it = self.get_dataset_iterator(image_id=image_id, category_id=category_id, split=split)
        category_dict = self.get_category_dict()

        result_dir = Path(result_dir or self.dataset_dir / "visualized").absolute()
        logger.info(f"Visualizing dataset [{self.name}] to {result_dir}")

        pgbar = tqdm(it, total=len(it), desc="Visualizing dataset")
        for data in pgbar:
            draw = visualize(
                image=data.image,
                annotations=data.annotations,
                category_dict=category_dict,
                image_info=data.image_info,
            )

            image_path = result_dir / (data.image_info.id + data.image_info.ext)
            image_io.cv2_imwrite(image_path, draw, create_directory=True)

            if show:
                image_io.cv2_imshow("vis", draw)

        logger.info(f"Visualizing dataset [{self.name}] to {result_dir} finished")

        return Path(result_dir)

    # @exception_decorator
    # def experimental_filter(
    #     self,
    #     new_dataset_name: str,
    #     image_lambda: callable[ImageInfo, bool] = None,
    #     annotation_lambda: callable[AnnotationInfo, bool] = None,
    #     category_lambda: callable[CategoryInfo, bool] = None,
    #     root_dir: Union[str, Path] = None,
    # ) -> "WaffleDataset":
    #     pass

    # @exception_decorator
    # def experimental_manipulate(
    #     self,
    #     new_dataset_name: str,
    #     task: Union[str, TaskType],
    #     image_lambda: callable[ImageInfo, bool] = None,
    #     annotation_lambda: callable[AnnotationInfo, bool] = None,
    #     category_lambda: callable[CategoryInfo, bool] = None,
    #     root_dir: Union[str, Path] = None,
    # ) -> "WaffleDataset":
    #     pass
