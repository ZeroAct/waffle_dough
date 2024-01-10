import cv2
import numpy as np
import pytest
import sqlalchemy

from waffle_dough.dataset.dataset import WaffleDataset
from waffle_dough.exception.database_exception import *
from waffle_dough.exception.dataset_exception import *
from waffle_dough.field import AnnotationInfo, CategoryInfo, ImageInfo
from waffle_dough.type import SplitType, TaskType


def test_waffle_dataset_new(tmpdir):
    dataset_list = WaffleDataset.get_dataset_list(root_dir=tmpdir)
    assert dataset_list == []

    dataset = WaffleDataset.new("test1", task="classification", root_dir=tmpdir)
    dataset_info = dataset.get_dataset_info()
    assert dataset_info.name == "test1"
    assert dataset_info.task == "classification"

    with pytest.raises(DatasetAlreadyExistsError):
        WaffleDataset.new("test1", task="classification", root_dir=tmpdir)

    dataset_list = WaffleDataset.get_dataset_list(root_dir=tmpdir)
    assert dataset_list == ["test1"]

    dataset_list = WaffleDataset.get_dataset_list(root_dir=tmpdir, task="classification")
    assert dataset_list == ["test1"]

    dataset_list = WaffleDataset.get_dataset_list(root_dir=tmpdir, task="CLASSIFICATION")
    assert dataset_list == ["test1"]

    dataset_list = WaffleDataset.get_dataset_list(root_dir=tmpdir, task=TaskType.CLASSIFICATION)
    assert dataset_list == ["test1"]

    dataset_list = WaffleDataset.get_dataset_list(root_dir=tmpdir, task="object_detection")
    assert dataset_list == []

    dataset = WaffleDataset.new("test2", task="object_detection", root_dir=tmpdir)
    dataset_list = WaffleDataset.get_dataset_list(root_dir=tmpdir)
    assert dataset_list == ["test1", "test2"]

    dataset_list = WaffleDataset.get_dataset_list(root_dir=tmpdir, task="object_detection")
    assert dataset_list == ["test2"]


def test_waffle_dataset_load(tmpdir):
    dataset = WaffleDataset.new("test1", task="classification", root_dir=tmpdir)

    dataset = WaffleDataset.load("test1", root_dir=tmpdir)
    dataset_info = dataset.get_dataset_info()
    assert dataset_info.name == "test1"
    assert dataset_info.task == "classification"

    with pytest.raises(DatasetNotFoundError):
        WaffleDataset.load("test2", root_dir=tmpdir)


def test_waffle_dataset_delete(tmpdir):
    dataset = WaffleDataset.new("test1", task="classification", root_dir=tmpdir)
    dataset_list = WaffleDataset.get_dataset_list(root_dir=tmpdir)
    assert dataset_list == ["test1"]

    WaffleDataset.delete("test1", root_dir=tmpdir)
    dataset_list = WaffleDataset.get_dataset_list(root_dir=tmpdir)
    assert dataset_list == []

    with pytest.raises(DatasetNotFoundError):
        WaffleDataset.delete("test1", root_dir=tmpdir)


def test_waffle_dataset_copy(tmpdir):
    dataset = WaffleDataset.new("test1", task="classification", root_dir=tmpdir)
    dataset_list = WaffleDataset.get_dataset_list(root_dir=tmpdir)
    assert dataset_list == ["test1"]

    dataset = WaffleDataset.copy("test1", "test2", root_dir=tmpdir)
    dataset_list = WaffleDataset.get_dataset_list(root_dir=tmpdir)
    assert dataset_list == ["test1", "test2"]

    with pytest.raises(DatasetAlreadyExistsError):
        WaffleDataset.copy("test1", "test2", root_dir=tmpdir)

    with pytest.raises(DatasetNotFoundError):
        WaffleDataset.copy("test3", "test4", root_dir=tmpdir)


def test_waffle_dataset_crud(tmpdir, sample_image_paths):
    dataset = WaffleDataset.new("test1", task="classification", root_dir=tmpdir)

    # image
    dataset.add_image(sample_image_paths[0])
    assert len(dataset.get_images()) == len(dataset.get_image_dict()) == 1

    dataset.add_image(sample_image_paths[1])
    assert len(dataset.get_images()) == len(dataset.get_image_dict()) == 2

    image_id = list(dataset.get_image_dict().keys())[0]
    dataset.delete_image(image_id)
    assert len(dataset.get_images()) == len(dataset.get_image_dict()) == 1

    # category
    dataset.add_category(category_info=CategoryInfo.classification(name="test1"))
    assert len(dataset.get_categories()) == len(dataset.get_category_dict()) == 1

    dataset.add_category(category_info={"name": "test2"})
    assert len(dataset.get_categories()) == len(dataset.get_category_dict()) == 2

    with pytest.raises(BaseException):
        dataset.add_category(category_info=CategoryInfo.classification(name="test1"))

    category_id = list(dataset.get_category_dict().keys())[0]
    dataset.delete_category(category_id)
    assert len(dataset.get_categories()) == len(dataset.get_category_dict()) == 1

    # annotation
    image_id = list(dataset.get_image_dict().keys())[0]
    category_id = list(dataset.get_category_dict().keys())[0]

    dataset.add_annotation(
        AnnotationInfo.classification(
            image_id=image_id,
            category_id=category_id,
        )
    )
    assert len(dataset.get_annotations()) == len(dataset.get_annotation_dict()) == 1

    with pytest.raises(DatabaseNotFoundError):
        dataset.add_annotation(
            AnnotationInfo.classification(
                image_id="random_id",
                category_id=category_id,
            )
        )

    with pytest.raises(DatabaseNotFoundError):
        dataset.add_annotation(
            AnnotationInfo.classification(
                image_id=image_id,
                category_id="random_id",
            )
        )

    with pytest.raises(DatasetTaskError):
        dataset.add_annotation(
            AnnotationInfo.object_detection(
                image_id=image_id,
                category_id=category_id,
                bbox=[0, 0, 1.0, 1.0],
            )
        )

    # cascade delete
    dataset = WaffleDataset.new("test2", task="classification", root_dir=tmpdir)
    images = dataset.add_image(sample_image_paths[-2:])
    category = dataset.add_category(category_info=CategoryInfo.classification(name="test"))[0]

    annotations = dataset.add_annotation(
        [
            AnnotationInfo.classification(
                image_id=image.id,
                category_id=category.id,
            )
            for image in images
        ]
    )
    assert len(dataset.get_annotation_dict()) == 2

    dataset.delete_image(images[0].id)
    assert len(dataset.get_annotation_dict()) == 1

    dataset.delete_category(category.id)
    assert len(dataset.get_annotation_dict()) == 0


def test_waffle_dataset_split(tmpdir, sample_image_paths):
    dataset = WaffleDataset.new("test1", task="classification", root_dir=tmpdir)

    with pytest.raises(DatasetEmptyError):
        dataset.random_split(0.5)

    images = dataset.add_image(sample_image_paths)
    with pytest.raises(DatasetEmptyError):
        dataset.random_split(0.8, 0.1, 0.1)

    category = dataset.add_category(category_info=CategoryInfo.classification(name="test"))[0]
    dataset.add_annotation(
        annotation_info=[
            AnnotationInfo.classification(
                image_id=image.id,
                category_id=category.id,
            )
            for image in images
        ]
    )

    with pytest.raises(DatasetSplitError):
        dataset.random_split(-1)

    with pytest.raises(DatasetSplitError):
        dataset.random_split(0, 0, 0)

    image_num = len(sample_image_paths)
    dataset.random_split(0.8, 0.1, 0.1)
    assert len(dataset.get_images()) == image_num
    assert len(dataset.get_images(split=SplitType.TRAIN)) == round(image_num * 0.8)
    assert len(dataset.get_images(split=SplitType.VALIDATION)) == round(image_num * 0.1)
    assert len(dataset.get_images(split=SplitType.TEST)) == round(image_num * 0.1)


def test_waffle_dataset_get_dataset_iterator(tmpdir, sample_image_paths):
    dataset = WaffleDataset.new("test1", task="classification", root_dir=tmpdir)

    it = dataset.get_dataset_iterator()
    assert len(it) == 0

    images = dataset.add_image(sample_image_paths)
    assert len(dataset.get_dataset_iterator()) == len(images)
    assert len(dataset.get_dataset_iterator(split=SplitType.TRAIN)) == 0
    assert len(dataset.get_dataset_iterator(split=SplitType.UNSET)) == len(images)

    category = dataset.add_category(category_info=CategoryInfo.classification(name="test"))[0]
    dataset.add_annotation(
        annotation_info=[
            AnnotationInfo.classification(
                image_id=image.id,
                category_id=category.id,
            )
            for image in images
        ]
    )
    dataset.random_split(0.8, 0.1, 0.1)

    assert len(dataset.get_dataset_iterator(split=SplitType.TRAIN)) == round(len(images) * 0.8)
    assert len(dataset.get_dataset_iterator(split=SplitType.VALIDATION)) == round(len(images) * 0.1)
    assert len(dataset.get_dataset_iterator(split=SplitType.TEST)) == round(len(images) * 0.1)

    it = dataset.get_dataset_iterator(split=SplitType.TRAIN)
    assert hasattr(it[0], "image") and isinstance(it[0].image, np.ndarray)
    assert hasattr(it[0], "image_path")
    assert hasattr(it[0], "image_info")
    assert hasattr(it[0], "annotations")
    assert hasattr(it[0], "categories")


def test_waffle_dataset_visualize(tmpdir, sample_image_paths):
    dataset = WaffleDataset.new("test1", task="classification", root_dir=tmpdir)

    images = dataset.add_image(sample_image_paths)
    draw_dir = dataset.visualize()
    assert draw_dir.exists() and len(list(draw_dir.iterdir())) == len(images)

    category = dataset.add_category(category_info=CategoryInfo.classification(name="test"))[0]
    dataset.add_annotation(
        annotation_info=[
            AnnotationInfo.classification(
                image_id=image.id,
                category_id=category.id,
            )
            for image in images
        ]
    )
    dataset.random_split(0.8, 0.1, 0.1)

    draw_dir = dataset.visualize(result_dir=tmpdir / "draw", split=SplitType.TRAIN)
    assert draw_dir.exists() and len(list(draw_dir.iterdir())) == round(len(images) * 0.8)


# convert
def test_waffle_dataset_from_coco(tmpdir, sample_image_paths):
    coco = {
        "categories": [
            {"id": 1, "name": "dog"},
            {"id": 2, "name": "cat"},
        ],
        "images": [
            {"id": 1, "file_name": "dog.png", "width": 300, "height": 300},
            {"id": 2, "file_name": "cat.png", "width": 300, "height": 300},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [50, 50, 50, 50]},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [80, 80, 100, 100]},
            {"id": 3, "image_id": 2, "category_id": 2, "bbox": [120, 120, 100, 100]},
        ],
    }

    image_dir = tmpdir / "coco_images"
    image_dir.mkdir()
    for image in coco["images"]:
        img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        cv2.imwrite(str(image_dir / image["file_name"]), img)

    dataset = WaffleDataset.from_coco(
        name="coco_import",
        task="object_detection",
        coco=coco,
        coco_image_dir=image_dir,
        root_dir=tmpdir,
    )

    dataset.visualize(result_dir="/home/zero/ws/waffle_dough/vis")
