import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

from waffle_utils.file import io

from waffle_dough.exception.database_exception import *
from waffle_dough.field import (
    AnnotationInfo,
    CategoryInfo,
    ImageInfo,
    UpdateAnnotationInfo,
    UpdateCategoryInfo,
    UpdateImageInfo,
)
from waffle_dough.type.split_type import SplitType

from .engine import create_session
from .repository import (
    annotation_repository,
    category_repository,
    image_repository,
)

logger = logging.getLogger(__name__)


class DatabaseService:
    def __init__(self, db_url: str, image_directory: Union[str, Path]):
        self.Session = create_session(db_url)
        self.image_directory = Path(image_directory)

    # Create
    def add_image(self, image: str, image_info: ImageInfo) -> ImageInfo:
        with self.Session() as session:
            io.copy_file(image, self.get_image_path(image_info), create_directory=True)
            try:
                image_info = image_repository.create(session, image_info)
            except Exception as e:
                io.remove_file(self.get_image_path(image_info))
                raise e

        return ImageInfo.model_validate(image_info, from_attributes=True)

    def add_category(self, category_info: CategoryInfo) -> CategoryInfo:
        with self.Session() as session:
            category = category_repository.create(session, category_info)

        return CategoryInfo.model_validate(category, from_attributes=True)

    def add_annotation(self, annotation_info: AnnotationInfo) -> AnnotationInfo:
        with self.Session() as session:
            annotation = annotation_repository.create(session, annotation_info)

        return AnnotationInfo.model_validate(annotation, from_attributes=True)

    # Read (Unit)
    def get_image(self, image_id: str) -> ImageInfo:
        with self.Session() as session:
            image = image_repository.get(session, image_id)

        if image is None:
            raise DatabaseNotFoundError(f"image does not exist: {image_id}")

        return ImageInfo.model_validate(image, from_attributes=True)

    def get_image_path(self, image_info: ImageInfo) -> Path:
        return Path(self.image_directory, f"{image_info.id}{image_info.ext}")

    def get_category(self, category_id: str) -> CategoryInfo:
        with self.Session() as session:
            category = category_repository.get(session, category_id)

        if category is None:
            raise DatabaseNotFoundError(f"category does not exist: {category_id}")

        return CategoryInfo.model_validate(category, from_attributes=True)

    def get_annotation(self, annotation_id: str) -> AnnotationInfo:
        with self.Session() as session:
            annotation = annotation_repository.get(session, annotation_id)

        if annotation is None:
            raise DatabaseNotFoundError(f"annotation does not exist: {annotation_id}")

        return AnnotationInfo.model_validate(annotation, from_attributes=True)

    # Read (Multi)
    def _get_query_dict(
        self,
        id: Union[str, list[str]] = None,
        image_id: Union[str, list[str]] = None,
        category_id: Union[str, list[str]] = None,
        annotation_id: Union[str, list[str]] = None,
        split: SplitType = None,
        filter_by: dict[str, Any] = {},
        filter_in: dict[str, Optional[list]] = {},
        filter_like: list[Tuple[str, str]] = [],
        order_by: List[Tuple[str, str]] = [],
        skip: int = None,
        limit: int = None,
    ) -> dict[str, Any]:
        query_dict = {}

        filter_by = filter_by.copy()
        filter_in = filter_in.copy()
        filter_like = filter_like.copy()
        order_by = order_by.copy()

        if id:
            filter_in.update({"id": [id] if isinstance(id, str) else id})

        if image_id:
            filter_in.update({"image_id": [image_id] if isinstance(image_id, str) else image_id})

        if category_id:
            filter_in.update(
                {"category_id": [category_id] if isinstance(category_id, str) else category_id}
            )

        if annotation_id:
            filter_in.update(
                {"id": [annotation_id] if isinstance(annotation_id, str) else annotation_id}
            )

        if split:
            filter_by.update({"split": split})

        query_dict.update(
            {
                "filter_by": filter_by,
                "filter_in": filter_in,
                "filter_like": filter_like,
                "order_by": order_by,
                "skip": skip,
                "limit": limit,
            }
        )

        return query_dict

    def _get_multi(self, repository, **kwargs) -> List[Any]:
        with self.Session() as session:
            objs = repository.get_multi(session, **self._get_query_dict(**kwargs))

        return objs

    def _get_count(self, repository, **kwargs) -> int:
        with self.Session() as session:
            count = repository.get_count(session, **self._get_query_dict(**kwargs))

        return count

    def get_images(
        self, image_id: Union[str, list[str]] = None, split: SplitType = None, **kwargs
    ) -> dict[str, ImageInfo]:
        images = self._get_multi(image_repository, id=image_id, split=split, **kwargs)
        return {image.id: ImageInfo.model_validate(image, from_attributes=True) for image in images}

    def get_categories(
        self, category_id: Union[str, list[str]] = None, **kwargs
    ) -> dict[str, CategoryInfo]:
        categories = self._get_multi(category_repository, id=category_id, **kwargs)
        return {
            category.id: CategoryInfo.model_validate(category, from_attributes=True)
            for category in categories
        }

    def get_annotations(
        self, annotation_id: Union[str, list[str]] = None, **kwargs
    ) -> dict[str, AnnotationInfo]:
        annotations = self._get_multi(annotation_repository, id=annotation_id, **kwargs)
        return {
            annotation.id: AnnotationInfo.model_validate(annotation, from_attributes=True)
            for annotation in annotations
        }

    # Read with relations
    def get_images_by_category_id(
        self, category_id: Union[str, list[str]], split: SplitType = None
    ) -> dict[str, ImageInfo]:
        annotations = self._get_multi(annotation_repository, category_id=category_id)
        image_ids = list(set(map(lambda annotation: annotation.image_id, annotations)))
        images = self._get_multi(image_repository, id=image_ids, split=split)
        return {image.id: ImageInfo.model_validate(image, from_attributes=True) for image in images}

    def get_images_by_annotation_id(
        self, annotation_id: Union[str, list[str]], split: SplitType = None
    ) -> dict[str, ImageInfo]:
        annotations = self._get_multi(annotation_repository, annotation_id=annotation_id)
        image_ids = list(set(map(lambda annotation: annotation.image_id, annotations)))
        images = self._get_multi(image_repository, id=image_ids, split=split)
        return {image.id: ImageInfo.model_validate(image, from_attributes=True) for image in images}

    def get_categories_by_annotation_id(
        self, annotation_id: Union[str, list[str]]
    ) -> dict[str, CategoryInfo]:
        annotations = self._get_multi(annotation_repository, annotation_id=annotation_id)
        category_ids = list(set(map(lambda annotation: annotation.category_id, annotations)))
        categories = self._get_multi(category_repository, id=category_ids)
        return {
            category.id: CategoryInfo.model_validate(category, from_attributes=True)
            for category in categories
        }

    def get_categories_by_image_id(self, image_id: Union[str, list[str]]) -> dict[str, CategoryInfo]:
        annotations = self._get_multi(annotation_repository, image_id=image_id)
        category_ids = list(set(map(lambda annotation: annotation.category_id, annotations)))
        categories = self._get_multi(category_repository, id=category_ids)
        return {
            category.id: CategoryInfo.model_validate(category, from_attributes=True)
            for category in categories
        }

    def get_annotations_by_image_id(
        self, image_id: Union[str, list[str]]
    ) -> dict[str, AnnotationInfo]:
        annotations = self._get_multi(annotation_repository, image_id=image_id)
        return {
            annotation.id: AnnotationInfo.model_validate(annotation, from_attributes=True)
            for annotation in annotations
        }

    def get_annotations_by_category_id(
        self, category_id: Union[str, list[str]]
    ) -> dict[str, AnnotationInfo]:
        annotations = self._get_multi(annotation_repository, category_id=category_id)
        return {
            annotation.id: AnnotationInfo.model_validate(annotation, from_attributes=True)
            for annotation in annotations
        }

    # Read custom
    def get_image_count(self, split: SplitType = None, **kwargs) -> int:
        return self._get_count(image_repository, split=split, **kwargs)

    def get_category_count(self) -> int:
        return self._get_count(category_repository)

    def get_annotation_count(self) -> int:
        return self._get_count(annotation_repository)

    def get_image_by_original_file_name(self, original_file_name: str) -> ImageInfo:
        with self.Session() as session:
            image = image_repository.get_multi(
                session, filter_by={"original_file_name": original_file_name}
            )

        if len(image) == 0:
            raise DatabaseNotFoundError(f"image does not exist: {original_file_name}")
        elif len(image) > 1:
            raise DatabaseIntegrityError(f"multiple images exist: {original_file_name}")

        return ImageInfo.model_validate(image[0], from_attributes=True)

    def get_category_by_name(self, name: str) -> CategoryInfo:
        with self.Session() as session:
            category = category_repository.get_multi(session, filter_by={"name": name})

        if len(category) == 0:
            raise DatabaseNotFoundError(f"category does not exist: {name}")
        elif len(category) > 1:
            raise DatabaseIntegrityError(f"multiple categories exist: {name}")

        return CategoryInfo.model_validate(category[0], from_attributes=True)

    # stats
    def get_image_num_by_category_id(
        self, category_id: Union[str, list[str]] = None
    ) -> dict[str, int]:
        annotations = self._get_multi(annotation_repository, category_id=category_id)

        image_num_by_category_id = {}
        for annotation in annotations:
            if annotation.category_id not in image_num_by_category_id:
                image_num_by_category_id[annotation.category_id] = 0
            image_num_by_category_id[annotation.category_id] += 1

        return image_num_by_category_id

    def get_annotation_num_by_category_id(
        self, category_id: Union[str, list[str]] = None
    ) -> dict[str, int]:
        annotations = self._get_multi(annotation_repository, category_id=category_id)

        annotation_num_by_category_id = {}
        for annotation in annotations:
            if annotation.category_id not in annotation_num_by_category_id:
                annotation_num_by_category_id[annotation.category_id] = 0
            annotation_num_by_category_id[annotation.category_id] += 1

        return annotation_num_by_category_id

    def get_annotation_num_by_image_id(
        self, image_id: Union[str, list[str]] = None
    ) -> dict[str, int]:
        annotations = self._get_multi(annotation_repository, image_id=image_id)

        annotation_num_by_image_id = {}
        for annotation in annotations:
            if annotation.image_id not in annotation_num_by_image_id:
                annotation_num_by_image_id[annotation.image_id] = 0
            annotation_num_by_image_id[annotation.image_id] += 1

        return annotation_num_by_image_id

    # Update
    def update_image(self, image_id: str, update_image_info: UpdateImageInfo) -> ImageInfo:
        with self.Session() as session:
            image = image_repository.update(session, image_id, update_image_info)

        return ImageInfo.model_validate(image, from_attributes=True)

    def update_category(
        self, category_id: str, update_category_info: UpdateCategoryInfo
    ) -> CategoryInfo:
        with self.Session() as session:
            category = category_repository.update(session, category_id, update_category_info)

        return CategoryInfo.model_validate(category, from_attributes=True)

    def update_annotation(
        self, annotation_id: str, update_annotation_info: UpdateAnnotationInfo
    ) -> AnnotationInfo:
        with self.Session() as session:
            annotation = annotation_repository.update(session, annotation_id, update_annotation_info)

        return AnnotationInfo.model_validate(annotation, from_attributes=True)

    # Delete
    def delete_image(self, image_id: str) -> None:
        with self.Session() as session:
            image_info = image_repository.remove(session, image_id)

        image_path = self.get_image_path(image_info)
        if Path(image_path).exists():
            io.remove_file(image_path)

    def delete_category(self, category_id: str) -> None:
        with self.Session() as session:
            category_repository.remove(session, category_id)

    def delete_annotation(self, annotation_id: str) -> None:
        with self.Session() as session:
            annotation_repository.remove(session, annotation_id)
