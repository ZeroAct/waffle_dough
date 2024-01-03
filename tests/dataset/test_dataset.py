import pytest
import sqlalchemy

from waffle_dough.dataset.dataset import WaffleDataset
from waffle_dough.exception.database_exception import *
from waffle_dough.exception.dataset_exception import *
from waffle_dough.field import AnnotationInfo, CategoryInfo, ImageInfo
from waffle_dough.type import TaskType


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

    with pytest.raises(sqlalchemy.exc.IntegrityError):
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
