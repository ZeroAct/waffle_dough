from pathlib import Path

import cv2
import numpy as np
import pytest
from waffle_utils.file import io, network

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

    dataset.add_image(cv2.imread(sample_image_paths[1]))
    assert len(dataset.get_images()) == len(dataset.get_image_dict()) == 2

    image_id = list(dataset.get_image_dict().keys())[0]
    dataset.delete_image(image_id)
    assert len(dataset.get_images()) == len(dataset.get_image_dict()) == 1

    # category
    category = dataset.add_category(category_info=CategoryInfo.classification(name="test1"))[0]
    assert len(dataset.get_categories()) == len(dataset.get_category_dict()) == 1

    dataset_info = dataset.get_dataset_info()
    assert len(dataset_info.categories) == 1
    assert dataset_info.categories[0]["name"] == "test1"

    dataset.update_category(category.id, CategoryInfo.classification(name="test1_update"))
    assert dataset.categories[0].name == "test1_update"
    dataset_info = dataset.get_dataset_info()
    assert len(dataset_info.categories) == 1
    assert dataset_info.categories[0]["name"] == "test1_update"

    dataset.add_category(category_info={"name": "test2"})
    assert len(dataset.get_categories()) == len(dataset.get_category_dict()) == 2

    with pytest.raises(BaseException):
        dataset.add_category(category_info=CategoryInfo.classification(name="test2"))

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
@pytest.mark.parametrize(
    "file_name, task",
    [
        ("mnist.zip", "object_detection"),
        ("mnist.zip", "semantic_segmentation"),
        ("mnist.zip", "instance_segmentation"),
    ],
)
def test_waffle_dataset_coco(tmpdir, file_name, task):
    dataset_url = f"https://github.com/snuailab/assets/raw/main/waffle/sample_dataset/{file_name}"
    dataset_path = tmpdir / file_name
    coco_dir = tmpdir / Path(file_name).stem

    network.get_file_from_url(dataset_url, dataset_path)
    io.unzip(dataset_path, coco_dir, create_directory=True)

    dataset1 = WaffleDataset.new("test1", task=task, root_dir=tmpdir)
    dataset1.import_coco(coco=coco_dir / "coco.json", coco_image_dir=coco_dir / "images")
    dataset1.visualize()
    export_dir = dataset1.export("coco")
    assert Path(export_dir).exists()

    dataset2 = WaffleDataset.new("test2", task=task, root_dir=tmpdir)
    for coco_file in Path(export_dir).glob("*.json"):
        dataset2.import_coco(coco=coco_file, coco_image_dir=Path(export_dir) / "images")
    dataset2.visualize()

    assert len(dataset1.get_images()) == len(dataset2.get_images())
    assert len(dataset1.get_categories()) == len(dataset2.get_categories())
    assert len(dataset1.get_annotations()) == len(dataset2.get_annotations())


@pytest.mark.parametrize(
    "file_name, task",
    [
        ("mnist_yolo_object_detection.zip", "object_detection"),
        ("mnist_yolo_classification.zip", "classification"),
        ("mnist_yolo_instance_segmentation.zip", "instance_segmentation"),
    ],
)
def test_waffle_dataset_yolo(tmpdir, file_name, task):
    dataset_url = f"https://github.com/snuailab/assets/raw/main/waffle/sample_dataset/{file_name}"
    dataset_path = tmpdir / file_name
    yolo_dir = tmpdir / Path(file_name).stem

    network.get_file_from_url(dataset_url, dataset_path)
    io.unzip(dataset_path, yolo_dir, create_directory=True)

    dataset1 = WaffleDataset.new("test1", task=task, root_dir=tmpdir)
    dataset1.import_yolo(yolo_root_dir=yolo_dir)
    dataset1.visualize()
    export_dir = dataset1.export("yolo")
    assert Path(export_dir).exists()

    dataset2 = WaffleDataset.new("test2", task=task, root_dir=tmpdir)
    dataset2.import_yolo(yolo_root_dir=export_dir)
    dataset2.visualize()

    assert len(dataset1.get_images()) == len(dataset2.get_images())
    assert len(dataset1.get_categories()) == len(dataset2.get_categories())
    assert len(dataset1.get_annotations()) == len(dataset2.get_annotations())


def test_waffle_dataset_statistic(tmpdir, sample_image_paths):
    dataset = WaffleDataset.new("test1", task="classification", root_dir=tmpdir)
    stat = dataset.get_statistics()
    assert stat.num_images == stat.num_annotations == stat.num_categories == 0

    images = dataset.add_image(sample_image_paths)
    stat = dataset.get_statistics()
    assert stat.num_images == len(sample_image_paths)

    category1 = dataset.add_category(category_info=CategoryInfo.classification(name="test"))[0]
    stat = dataset.get_statistics()
    assert stat.num_categories == 1

    dataset.add_annotation(
        annotation_info=[
            AnnotationInfo.classification(
                image_id=image.id,
                category_id=category1.id,
            )
            for image in images
        ]
    )
    stat = dataset.get_statistics()
    assert stat.num_annotations == len(sample_image_paths)

    category2 = dataset.add_category(category_info=CategoryInfo.classification(name="test2"))[0]
    stat = dataset.get_statistics()
    assert stat.num_categories == 2
    assert stat.num_annotations_per_category[category1.name] == len(sample_image_paths)
    assert stat.num_annotations_per_category[category2.name] == 0
