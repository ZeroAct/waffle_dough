import pytest

from waffle_dough.field import AnnotationInfo, CategoryInfo, ImageInfo
from waffle_dough.type import TaskType


@pytest.mark.parametrize(
    "task, kwargs",
    [
        (TaskType.CLASSIFICATION, {"annotation_id": 1, "image_id": 1, "category_id": 1}),
        ("classification", {"annotation_id": 1, "image_id": 1, "category_id": 1}),
        ("CLAssificatioN", {"annotation_id": 1, "image_id": 1, "category_id": 1}),
        (
            TaskType.OBJECT_DETECTION,
            {"annotation_id": 1, "image_id": 1, "bbox": [100, 100, 100, 100], "category_id": 1},
        ),
        (
            TaskType.SEMANTIC_SEGMENTATION,
            {
                "annotation_id": 1,
                "image_id": 1,
                "segmentation": [[110, 110, 130, 130, 110, 130]],
                "category_id": 1,
            },
        ),
        (
            TaskType.INSTANCE_SEGMENTATION,
            {
                "annotation_id": 1,
                "image_id": 1,
                "segmentation": [[110, 110, 130, 130, 110, 130]],
                "category_id": 1,
            },
        ),
        (
            TaskType.KEYPOINT_DETECTION,
            {
                "annotation_id": 1,
                "image_id": 1,
                "keypoints": [0, 0, 0, 130, 130, 1, 110, 130, 2],
                "bbox": [100, 100, 100, 100],
                "num_keypoints": 3,
                "category_id": 1,
            },
        ),
        (
            TaskType.TEXT_RECOGNITION,
            {
                "annotation_id": 1,
                "image_id": 1,
                "caption": "1",
            },
        ),
        (
            TaskType.REGRESSION,
            {
                "annotation_id": 1,
                "image_id": 1,
                "value": 1,
            },
        ),
    ],
)
def test_annotation_info(task, kwargs):
    annotation_1 = AnnotationInfo.new(**kwargs, task=task)
    annotation_2 = getattr(AnnotationInfo, task.lower())(**kwargs)
    annotation_3 = AnnotationInfo.from_dict(kwargs, task=task)

    assert annotation_1 == annotation_2
    assert annotation_2 == annotation_3
    assert annotation_3 == annotation_1

    d = annotation_1.to_dict()
    temp_annotation = AnnotationInfo.from_dict(d, task=task)
    assert annotation_1 == temp_annotation

    annotation_1.score = 0.5
    assert annotation_1.is_prediction()

    annotation_1.score = None
    assert not annotation_1.is_prediction()


def test_image_info():

    image_1 = ImageInfo.new(
        image_id=1,
        file_name="test.jpg",
        width=100,
        height=100,
    )
    image_2 = ImageInfo.new(
        image_id=1,
        file_name="test.jpg",
        width=300,
        height=300,
        original_file_name="test_original.jpg",
    )
    image_3 = ImageInfo.new(
        image_id=1,
        file_name="test2.jpg",
        width=300,
        height=300,
        original_file_name="test_original2.jpg",
        date_captured="2021-01-01",
    )

    assert image_1 == image_2
    assert image_2 != image_3
    assert image_3 != image_1

    d = image_1.to_dict()
    temp_image = ImageInfo.from_dict(d)
    assert image_2 == temp_image


@pytest.mark.parametrize(
    "task, kwargs",
    [
        (TaskType.CLASSIFICATION, {"category_id": 1, "name": "test", "supercategory": "test"}),
        (TaskType.OBJECT_DETECTION, {"category_id": 1, "name": "test", "supercategory": "test"}),
        (
            TaskType.SEMANTIC_SEGMENTATION,
            {"category_id": 1, "name": "test", "supercategory": "test"},
        ),
        (
            TaskType.INSTANCE_SEGMENTATION,
            {"category_id": 1, "name": "test", "supercategory": "test"},
        ),
        (
            TaskType.KEYPOINT_DETECTION,
            {
                "category_id": 1,
                "name": "test",
                "supercategory": "test",
                "keypoints": ["a", "b", "c"],
                "skeleton": [[1, 2], [2, 3]],
            },
        ),
        (TaskType.TEXT_RECOGNITION, {"category_id": 1, "name": "test", "supercategory": "test"}),
    ],
)
def test_category_info(task, kwargs):
    category_1 = CategoryInfo.new(**kwargs, task=task)
    category_2 = getattr(CategoryInfo, task.lower())(**kwargs)
    category_3 = CategoryInfo.from_dict(kwargs, task=task)

    assert category_1 == category_2
    assert category_2 == category_3
    assert category_3 == category_1

    d = category_1.to_dict()
    temp_category = CategoryInfo.from_dict(d, task=task)
    assert category_1 == temp_category
