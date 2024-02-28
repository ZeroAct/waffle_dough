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
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Union

import numpy as np
from natsort import natsorted
from tqdm import tqdm
from waffle_utils.file import io
from waffle_utils.logger import datetime_now, initialize_logger

from waffle_dough.database.service import DatabaseService
from waffle_dough.dataset.adapter import COCOAdapter, YOLOAdapter
from waffle_dough.dataset.adapter.callback import (
    BaseDatasetAdapterCallback,
    DatasetAdapterFileProgressCallback,
    DatasetAdapterTqdmProgressCallback,
)
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
from waffle_dough.type import DataType, SplitType, TaskType

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


@dataclass
class DatasetStatistics:
    num_images: int
    num_annotations: int
    num_categories: int
    num_images_per_category: dict[str, int]
    num_annotations_per_category: dict[str, int]

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(dataset_statistics: dict):
        return DatasetStatistics(
            num_images=dataset_statistics["num_images"],
            num_annotations=dataset_statistics["num_annotations"],
            num_categories=dataset_statistics["num_categories"],
            num_images_per_category=dataset_statistics["num_images_per_category"],
            num_annotations_per_category=dataset_statistics["num_annotations_per_category"],
        )


class WaffleDataset:
    """WaffleDataset class

    WaffleDataset is a programmatic interface for managing datasets.
    It provide several methods such as data CRUD, format conversion, visualization, etc...

    Attributes:
        name (str): Name of the dataset
        task (str): Task of the dataset
        root_dir (str): Root directory of the dataset
        dataset_dir (Path): Directory of the dataset
        dataset_info_file (Path): File path of the dataset info
        image_dir (Path): Directory of the images
        database_file (Path): File path of the database
        export_dir (Path): Directory of the exported data
        log_dir (Path): Directory of the logs
        log_file (Path): File path of the log
        category_dict (dict[str, CategoryInfo]): Dictionary of the categories
        category_name_dict (dict[str, CategoryInfo]): Dictionary of the categories by name
        categories (list[CategoryInfo]): List of the categories
        category_names (list[str]): List of the category names
        image_dict (dict[str, ImageInfo]): Dictionary of the images
        images (list[ImageInfo]): List of the images
        annotation_dict (dict[str, AnnotationInfo]): Dictionary of the annotations
        annotations (list[AnnotationInfo]): List of the annotations

    """

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
            self.database_service = DatabaseService(
                str(self.database_file), image_directory=self.image_dir
            )
        else:
            if task is None:
                raise DatasetTaskError(f"Task is not specified")
            self.task = task
            self.initialize()
            self.database_service = DatabaseService(
                str(self.database_file), image_directory=self.image_dir
            )
            self.create_dataset_info()

    def __repr__(self) -> str:
        return f"WaffleDataset(name={self.name}, task={self.task}, root_dir={self.root_dir})"

    def __str__(self) -> str:
        return f"WaffleDataset(name={self.name}, task={self.task}, root_dir={self.root_dir})"

    def initialize(self):
        io.make_directory(self.dataset_dir)
        io.make_directory(self.image_dir)

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
    def export_dir(self) -> Path:
        return self.dataset_dir / "export"

    @property
    def log_dir(self) -> Path:
        return self.dataset_dir / "logs"

    @property
    def log_file(self) -> Path:
        return self.log_dir / "dataset.log"

    @property
    def category_dict(self) -> dict[str, CategoryInfo]:
        return self.database_service.get_categories()

    @property
    def category_name_dict(self) -> dict[str, CategoryInfo]:
        return {category.name: category for category in self.categories}

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

        self.database_service.add_category(category_infos)
        return category_infos

    # # @update_dataset_decorator
    @exception_decorator
    def add_image(
        self,
        image: Union[str, Path, np.ndarray, list[Union[str, Path, np.ndarray]]],
        image_info: Union[ImageInfo, list[ImageInfo], dict, list[dict]] = None,
    ) -> list[ImageInfo]:
        images = image if isinstance(image, list) else [image]

        if image_info is None:
            image_info = [None] * len(images)
        image_infos = image_info if isinstance(image_info, list) else [image_info]

        db_image_paths = []
        db_image_infos = []
        temp_files = []
        for i, (image, image_info) in enumerate(zip(images, image_infos)):
            temp_file = None
            if isinstance(image, np.ndarray):
                np_image = image
                original_file_name = f"waffle_{i}.png"
                temp_file = NamedTemporaryFile(suffix=".png")
                image_path = temp_file.name
                image_io.cv2_imwrite(image_path, np_image)
            else:
                np_image = image_io.cv2_imread(image)
                original_file_name = str(image)
                image_path = str(image)

            if isinstance(image_info, dict):
                image_info = ImageInfo.from_dict(task=TaskType.AGNOSTIC, d=image_info)
            elif isinstance(image_info, ImageInfo):
                pass
            elif image_info is None:
                image_info = ImageInfo.agnostic(
                    width=np_image.shape[1],
                    height=np_image.shape[0],
                    original_file_name=original_file_name,
                )
            else:
                raise DatasetCRUDError(f"Invalid image info: {image_info}")

            image_infos[i] = image_info

            db_image_paths.append(image_path)
            db_image_infos.append(image_info)
            temp_files.append(temp_file)

        self.database_service.add_image(db_image_paths, db_image_infos)

        for temp_file in temp_files:
            if temp_file is not None:
                temp_file.close()

        return image_infos

    # # @update_dataset_decorator
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

        self.database_service.add_annotation(annotation_infos)

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
    def get_statistics(self, split: Union[str, SplitType] = None) -> DatasetStatistics:
        num_annotations_per_category = {}
        num_images_per_category = {}

        category_dict = self.get_category_dict()
        for category in self.category_dict.values():
            num_annotations_per_category[category.name] = 0
            num_images_per_category[category.name] = set()

        for annotation in self.annotations:
            num_annotations_per_category[category_dict[annotation.category_id].name] += 1
            num_images_per_category[category_dict[annotation.category_id].name].add(
                annotation.image_id
            )

        num_images_per_category = {
            category_name: len(image_ids)
            for category_name, image_ids in num_images_per_category.items()
        }

        return DatasetStatistics(
            num_images=len(self.images),
            num_annotations=len(self.annotations),
            num_categories=len(self.categories),
            num_images_per_category=num_images_per_category,
            num_annotations_per_category=num_annotations_per_category,
        )

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
        annotation_dict = self.get_annotation_dict(
            image_id=image_id, category_id=category_id, split=split
        )

        mapper = {}

        if not labeled_only:
            for image_id, image in image_dict.items():
                if image.id not in mapper:
                    mapper[image.id] = {
                        "image_path": self.get_image_path(image),
                        "image_info": image,
                        "annotations": [],
                        "categories": [],
                    }

        for _, annotation in annotation_dict.items():
            image = image_dict[annotation.image_id]
            category = category_dict[annotation.category_id]

            if image.id not in mapper:
                mapper[image.id] = {
                    "image_path": self.get_image_path(image),
                    "image_info": image,
                    "annotations": [],
                    "categories": [],
                }
            mapper[image.id]["annotations"].append(annotation)
            mapper[image.id]["categories"].append(category)

        return mapper

    # @update_dataset_decorator
    @exception_decorator
    def update_image(
        self,
        image_id: Union[str, list[str]],
        update_image_info: Union[UpdateImageInfo, list[UpdateImageInfo]],
    ) -> list[ImageInfo]:
        return self.database_service.update_image(image_id, update_image_info)

    @update_dataset_decorator
    @exception_decorator
    def update_category(
        self,
        category_id: Union[str, list[str]],
        update_category_info: Union[UpdateCategoryInfo, list[UpdateCategoryInfo]],
    ) -> list[CategoryInfo]:
        return self.database_service.update_category(category_id, update_category_info)

    # @update_dataset_decorator
    @exception_decorator
    def update_annotation(
        self,
        annotation_id: Union[str, list[str]],
        update_annotation_info: Union[UpdateAnnotationInfo, list[UpdateAnnotationInfo]],
    ) -> list[AnnotationInfo]:
        return self.database_service.update_annotation(annotation_id, update_annotation_info)

    # @update_dataset_decorator
    @exception_decorator
    def delete_image(self, image_id: Union[str, list[str]]):
        self.database_service.delete_image(image_id)

    @update_dataset_decorator
    @exception_decorator
    def delete_category(self, category_id: Union[str, list[str]]):
        self.database_service.delete_category(category_id)

    # @update_dataset_decorator
    @exception_decorator
    def delete_annotation(self, annotation_id: Union[str, list[str]]):
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
        return natsorted(dataset_list)

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
        """Create a new dataset

        Args:
            name (str): Name of the dataset
            task (str): Task of the dataset. One of the TaskType.
            root_dir (str, optional): Root directory of the dataset

        Returns:
            WaffleDataset: Created dataset
        """
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
        """Load an existing dataset

        Args:
            name (str): Name of the dataset
            root_dir (str, optional): Root directory of the dataset

        Returns:
            WaffleDataset: Loaded dataset
        """
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
        """Delete an existing dataset

        Args:
            name (str): Name of the dataset
            root_dir (str, optional): Root directory of the dataset
        """
        dataset = WaffleDataset.load(name, root_dir=root_dir)
        del dataset.database_service
        io.remove_file(dataset.database_file)
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
        """Copy an existing dataset

        Args:
            src_name (str): Name of the source dataset
            dst_name (str): Name of the destination dataset
            root_dir (str, optional): Root directory of the dataset

        Returns:
            WaffleDataset: Copied dataset
        """
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

    @exception_decorator
    def import_data(
        self,
        image_dict: dict[str, ImageInfo],
        annotation_dict: dict[str, AnnotationInfo],
        category_dict: dict[str, CategoryInfo],
        image_dir: Union[str, Path],
        split: Union[str, SplitType] = None,
    ) -> "WaffleDataset":
        category_name_to_id = {cat.name: cat.id for cat in self.categories}

        new_filtered_categories = []
        new_id_to_old_id = {}
        for category in category_dict.values():
            if category.name in category_name_to_id:
                new_id_to_old_id[category.id] = category_name_to_id[category.name]
                category.id = category_name_to_id[category.name]
            else:
                new_filtered_categories.append(category)
        self.add_category(new_filtered_categories)

        image_path = []
        image_infos = []
        for image in image_dict.values():
            image_path.append(Path(image_dir, image.original_file_name))
            if split is not None:
                image.split = SplitType.from_str(split) if split else SplitType.UNSET
            image_infos.append(image)
        self.add_image(image_path, image_infos)

        new_annotations = []
        for annotation in annotation_dict.values():
            if annotation.category_id in new_id_to_old_id:
                annotation.category_id = new_id_to_old_id[annotation.category_id]
            new_annotations.append(annotation)
        self.add_annotation(new_annotations)

        return self

    @exception_decorator
    def import_waffle(
        self,
        name: str,
        split: Union[str, SplitType] = None,
        root_dir: Union[str, Path] = None,
    ) -> "WaffleDataset":
        """Import an existing Waffle dataset

        Args:
            name (str): Name of the source dataset
            split (str, optional): Split of the dataset
            root_dir (str, optional): Root directory of the dataset

        Returns:
            WaffleDataset: Imported dataset
        """
        source_dataset = WaffleDataset.load(name, root_dir=root_dir)

        self.import_data(
            image_dict=source_dataset.image_dict,
            annotation_dict=source_dataset.annotation_dict,
            category_dict=source_dataset.category_dict,
            image_dir=source_dataset.image_dir,
            split=split,
        )

        return self

    @exception_decorator
    def import_coco(
        self,
        coco: Union[str, Path, dict],
        coco_image_dir: Union[str, Path],
        split: Union[str, SplitType] = None,
        callbacks: list[BaseDatasetAdapterCallback] = None,
    ) -> "WaffleDataset":
        """Import COCO dataset

        Args:
            coco (str, Path, dict): COCO dataset
            coco_image_dir (str, Path): Directory of the COCO images
            split (str, optional): Split of the dataset
            callbacks (list[BaseDatasetAdapterCallback], optional): List of the callbacks

        Returns:
            WaffleDataset: Imported dataset
        """
        adapter = COCOAdapter(
            task=self.task,
            callbacks=[
                DatasetAdapterFileProgressCallback(file=self.log_dir / "from_coco.json"),
                DatasetAdapterTqdmProgressCallback(desc="Importing COCO dataset"),
            ]
            + (callbacks if callbacks else []),
        )

        adapter.import_target(coco)

        self.import_data(
            image_dict=adapter.image_dict,
            annotation_dict=adapter.annotation_dict,
            category_dict=adapter.category_dict,
            image_dir=coco_image_dir,
            split=split,
        )

        return self

    @exception_decorator
    def import_yolo(
        self,
        yolo_root_dir: Union[str, Path],
        split: Union[str, SplitType] = None,
        callbacks: list[BaseDatasetAdapterCallback] = None,
    ) -> "WaffleDataset":
        """Import YOLO dataset

        Args:
            yolo_root_dir (str, Path): Root directory of the YOLO dataset
            split (str, optional): Split of the dataset
            callbacks (list[BaseDatasetAdapterCallback], optional): List of the callbacks

        Returns:
            WaffleDataset: Imported dataset
        """
        adapter = YOLOAdapter(
            task=self.task,
            callbacks=[
                DatasetAdapterFileProgressCallback(file=self.log_dir / "from_yolo.json"),
                DatasetAdapterTqdmProgressCallback(desc="Importing YOLO dataset"),
            ]
            + (callbacks if callbacks else []),
        )

        adapter.import_target(yolo_root_dir)

        self.import_data(
            image_dict=adapter.image_dict,
            annotation_dict=adapter.annotation_dict,
            category_dict=adapter.category_dict,
            image_dir=yolo_root_dir,
            split=split,
        )

        return self

    @exception_decorator
    def export(
        self,
        data_type: Union[str, DataType],
        result_dir: Union[str, Path] = None,
        callbacks: list[BaseDatasetAdapterCallback] = None,
        force: bool = False,
    ) -> str:
        """Export dataset

        Args:
            data_type (str, DataType): Type of the exported data
            result_dir (str, Path, optional): Directory of the exported data
            callbacks (list[BaseDatasetAdapterCallback], optional): List of the callbacks
            force (bool, optional): Whether to force the export

        Returns:
            str: Directory of the exported data
        """
        result_dir = Path(result_dir or self.export_dir).absolute()
        if result_dir.exists():
            if not force:
                raise DatasetAdapterAlreadyExistsError(
                    f"Exported dataset already exists: {result_dir}"
                )
            logger.info(f"There is already a directory [{result_dir}]. Deleting...")
            io.remove_directory(result_dir, recursive=True)
        io.make_directory(result_dir)
        logger.info(f"Exporting dataset [{self.name}] to COCO format to {result_dir}")

        if data_type == DataType.COCO:
            adapter = COCOAdapter(
                task=self.task,
                image_dict=self.image_dict,
                annotation_dict=self.annotation_dict,
                category_dict=self.category_dict,
                callbacks=[
                    DatasetAdapterFileProgressCallback(file=self.log_dir / "export.json"),
                    DatasetAdapterTqdmProgressCallback(desc="Exporting dataset to COCO format"),
                ]
                + (callbacks if callbacks else []),
            )
            adapter.export_target(result_dir, self.image_dir)

        elif data_type == DataType.YOLO:
            adapter = YOLOAdapter(
                task=self.task,
                image_dict=self.image_dict,
                annotation_dict=self.annotation_dict,
                category_dict=self.category_dict,
                callbacks=[
                    DatasetAdapterFileProgressCallback(file=self.log_dir / "export.json"),
                    DatasetAdapterTqdmProgressCallback(desc="Exporting dataset to YOLO format"),
                ]
                + (callbacks if callbacks else []),
            )
            adapter.export_target(result_dir, image_path_getter=self.get_image_path)

        else:
            raise DatasetAdapterNotFoundError(
                f"Data type {data_type} is not supported yet. {list(DataType)}"
            )

        return str(result_dir)

    @exception_decorator
    def random_split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ):
        """Randomly split the dataset

        Args:
            train_ratio (float, optional): Ratio of the training set
            val_ratio (float, optional): Ratio of the validation set
            test_ratio (float, optional): Ratio of the test set
            seed (int, optional): Seed of the random number generator
        """
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
        old_image_ids = (
            list(self.get_image_dict(split=SplitType.TRAIN).keys())
            + list(self.get_image_dict(split=SplitType.VALIDATION).keys())
            + list(self.get_image_dict(split=SplitType.TEST).keys())
        )
        self.update_image(
            old_image_ids, [UpdateImageInfo(split=SplitType.UNSET)] * len(old_image_ids)
        )

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

        if train_image_ids:
            self.update_image(
                train_image_ids, [UpdateImageInfo(split=SplitType.TRAIN)] * len(train_image_ids)
            )
        if val_image_ids:
            self.update_image(
                val_image_ids, [UpdateImageInfo(split=SplitType.VALIDATION)] * len(val_image_ids)
            )
        if test_image_ids:
            self.update_image(
                test_image_ids, [UpdateImageInfo(split=SplitType.TEST)] * len(test_image_ids)
            )

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
        """Get dataset iterator

        Args:
            image_id (str, list[str], optional): ID of the image
            category_id (str, list[str], optional): ID of the category
            split (str, optional): Split of the dataset
            preprocess (callable, optional): Preprocess function
            augment (callable, optional): Augment function

        Returns:
            Iterator: Dataset iterator
        """
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
        """Visualize dataset

        Args:
            image_id (str, list[str], optional): ID of the image
            category_id (str, list[str], optional): ID of the category
            split (str, optional): Split of the dataset
            result_dir (str, Path, optional): Directory of the result
            show (bool, optional): Whether to show the visualization

        Returns:
            Path: Directory of the result
        """
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
