from pathlib import Path

import cv2
import numpy as np
import pytest

from waffle_dough.database.service import DatabaseService
from waffle_dough.field import (
    AnnotationInfo,
    CategoryInfo,
    ImageInfo,
    UpdateAnnotationInfo,
    UpdateCategoryInfo,
    UpdateImageInfo,
)


def gen_database_service(tmpdir):
    return DatabaseService(db_url=None, image_directory=tmpdir)


def gen_dump_image(image_path):
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    Path(image_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), image)

    return image


def add_sample_image(database_service, tmpdir):
    original_file_name = f"image_{len(tmpdir.listdir())}.png"
    image_path = tmpdir / original_file_name
    image = gen_dump_image(image_path)

    image_info = database_service.add_image(
        image_path,
        ImageInfo.agnostic(
            original_file_name=original_file_name,
            width=image.shape[1],
            height=image.shape[0],
        ),
    )
    return image_info


def add_sample_category(database_service):
    category_info = database_service.add_category(
        CategoryInfo.classification(
            name=f"test_{len(database_service.get_categories())}",
        )
    )
    return category_info


def test_image_crud(tmpdir):
    database_service = gen_database_service(tmpdir)

    image_info = add_sample_image(database_service, tmpdir)
    assert database_service.get_image_count() == 1
    original_file_name = image_info.original_file_name

    image_info = database_service.get_image(image_info.id)
    assert image_info.original_file_name == original_file_name

    image_info = database_service.get_image_by_original_file_name(original_file_name)
    assert image_info.original_file_name == original_file_name

    image_info = database_service.update_image(
        image_info.id,
        UpdateImageInfo(
            split="train",
        ),
    )
    assert image_info.split == "train"

    database_service.delete_image(image_info.id)
    assert database_service.get_image_count() == 0


def test_category_crud(tmpdir):
    database_service = gen_database_service(tmpdir)

    category_info = add_sample_category(database_service)
    assert database_service.get_category_count() == 1
    name = category_info.name

    category_info = database_service.get_category(category_info.id)
    assert category_info.name == name

    category_info = database_service.update_category(
        category_info.id,
        UpdateCategoryInfo(
            name="test2",
        ),
    )
    assert category_info.name == "test2"

    database_service.delete_category(category_info.id)
    assert database_service.get_category_count() == 0


def test_annotation_crud(tmpdir):
    database_service = gen_database_service(tmpdir)

    image_info = add_sample_image(database_service, tmpdir)
    category_info = add_sample_category(database_service)

    annotation_info = database_service.add_annotation(
        AnnotationInfo.classification(
            image_id=image_info.id,
            category_id=category_info.id,
        )
    )
    assert database_service.get_annotation_count() == 1

    annotation_info = database_service.get_annotation(annotation_info.id)
    assert annotation_info.image_id == image_info.id

    annotation_info = database_service.update_annotation(
        annotation_info.id,
        UpdateAnnotationInfo(
            category_id=None,
        ),
    )


def test_advance_read(tmpdir):
    database_service = gen_database_service(tmpdir)

    image_info1 = add_sample_image(database_service, tmpdir)
    image_info2 = add_sample_image(database_service, tmpdir)
    image_info3 = add_sample_image(database_service, tmpdir)

    category_info1 = add_sample_category(database_service)
    category_info2 = add_sample_category(database_service)

    annotation_info1 = database_service.add_annotation(
        AnnotationInfo.classification(
            image_id=image_info1.id,
            category_id=category_info1.id,
        )
    )
    annotation_info2 = database_service.add_annotation(
        AnnotationInfo.classification(
            image_id=image_info2.id,
            category_id=category_info2.id,
        )
    )
    annotation_info3 = database_service.add_annotation(
        AnnotationInfo.classification(
            image_id=image_info3.id,
            category_id=category_info1.id,
        )
    )

    # Test get all
    image_infos = database_service.get_images()
    assert len(image_infos) == 3

    category_infos = database_service.get_categories()
    assert len(category_infos) == 2

    annotation_infos = database_service.get_annotations()
    assert len(annotation_infos) == 3

    # Test get annotations
    annotation_infos = database_service.get_annotations_by_image_id(image_info1.id)
    assert len(annotation_infos) == 1

    annotation_infos = database_service.get_annotations_by_category_id(category_info1.id)
    assert len(annotation_infos) == 2

    # Test get images
    image_info = database_service.get_image_by_original_file_name(image_info1.original_file_name)
    assert image_info.id == image_info1.id

    image_infos = database_service.get_images_by_category_id(category_info1.id)
    assert len(image_infos) == 2

    image_infos = database_service.get_images_by_category_id(category_info2.id)
    assert len(image_infos) == 1

    image_infos = database_service.get_images_by_annotation_id(annotation_info1.id)
    assert len(image_infos) == 1

    # Test get categories
    category_info = database_service.get_category_by_name(category_info1.name)
    assert category_info.id == category_info1.id

    category_infos = database_service.get_categories_by_annotation_id(annotation_info1.id)
    assert len(category_infos) == 1

    category_infos = database_service.get_categories_by_image_id(image_info1.id)
    assert len(category_infos) == 1

    category_infos = database_service.get_categories_by_image_id(image_info2.id)
    assert len(category_infos) == 1

    # Test get statistics
    image_num_by_category_id = database_service.get_image_num_by_category_id()
    assert image_num_by_category_id == {category_info1.id: 2, category_info2.id: 1}

    annotation_num_by_image_id = database_service.get_annotation_num_by_image_id()
    assert annotation_num_by_image_id == {image_info1.id: 1, image_info2.id: 1, image_info3.id: 1}

    annotation_num_by_category_id = database_service.get_annotation_num_by_category_id()
    assert annotation_num_by_category_id == {category_info1.id: 2, category_info2.id: 1}

    # delete category
    database_service.delete_category(category_info1.id)
    assert database_service.get_category_count() == 1
    assert database_service.get_annotation_count() == 1
    assert database_service.get_image_count() == 3

    # get by split
    image_info = database_service.get_image(image_info1.id)
    database_service.update_image(
        image_info.id,
        UpdateImageInfo(
            split="train",
        ),
    )

    image_info = database_service.get_image(image_info2.id)
    database_service.update_image(
        image_info.id,
        UpdateImageInfo(
            split="validation",
        ),
    )

    images = database_service.get_images(split="train")
    assert len(images) == 1

    images = database_service.get_images(split="validation")
    assert len(images) == 1
