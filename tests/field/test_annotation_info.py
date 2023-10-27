import pytest

from waffle_dough.field import AnnotationInfo
from waffle_dough.type import TaskType


def _test_annotation_info(task, kwargs, expected_output):
    new_function = getattr(AnnotationInfo, task.lower())
    if not isinstance(expected_output, dict):
        with pytest.raises(expected_output):
            new_function(**kwargs)
    else:
        annotation_1 = new_function(**kwargs)
        output = annotation_1.to_dict()
        output.pop("id")
        assert output == expected_output

        annotation_2 = AnnotationInfo.from_dict(task=task, d=kwargs)
        assert annotation_1 == annotation_2

        kwargs.pop("task")
        kwargs.update({"image_id": annotation_1.image_id + "2"})
        annotation_3 = new_function(**kwargs)
        assert annotation_1 != annotation_3

        return annotation_1


@pytest.mark.parametrize(
    "task, kwargs, expected_output",
    [
        (
            TaskType.CLASSIFICATION,
            {"image_id": "test", "category_id": "test"},
            {
                "task": "classification",
                "image_id": "test",
                "category_id": "test",
                "is_prediction": False,
            },
        ),
        (
            TaskType.CLASSIFICATION,
            {"category_id": "test"},
            TypeError,
        ),
        (
            TaskType.CLASSIFICATION,
            {"image_id": None, "category_id": "test"},
            ValueError,
        ),
    ],
)
def test_classification_annotation_info(task, kwargs, expected_output):
    _test_annotation_info(task, kwargs, expected_output)


@pytest.mark.parametrize(
    "task, kwargs, expected_output",
    [
        (
            TaskType.OBJECT_DETECTION,
            {"image_id": "test", "category_id": "test", "bbox": [0, 0, 1, 1]},
            {
                "task": "object_detection",
                "image_id": "test",
                "category_id": "test",
                "bbox": [0, 0, 1, 1],
                "area": 1,
                "iscrowd": 0,
                "is_prediction": False,
            },
        ),
        (
            TaskType.OBJECT_DETECTION,
            {"category_id": "test", "bbox": [0, 0, 1, 1]},
            TypeError,
        ),
        (
            TaskType.OBJECT_DETECTION,
            {"image_id": None, "category_id": "test", "bbox": [0, 0, 1, 1]},
            ValueError,
        ),
        (
            TaskType.OBJECT_DETECTION,
            {"image_id": "test", "bbox": [0, 0, 1, 1]},
            TypeError,
        ),
        (
            TaskType.OBJECT_DETECTION,
            {"image_id": "test", "category_id": "test", "bbox": [0, 0, 1]},
            ValueError,
        ),
        (
            TaskType.OBJECT_DETECTION,
            {"image_id": "test", "category_id": "test", "bbox": [0, 0, 1, 1, 1]},
            ValueError,
        ),
        (
            TaskType.OBJECT_DETECTION,
            {"image_id": "test", "category_id": "test", "bbox": [0, 0, 1, 1], "area": "asdf"},
            ValueError,
        ),
    ],
)
def test_object_detection_annotation_info(task, kwargs, expected_output):
    _test_annotation_info(task, kwargs, expected_output)


@pytest.mark.parametrize(
    "task, kwargs, expected_output",
    [
        (
            TaskType.SEMANTIC_SEGMENTATION,
            {"image_id": "test", "category_id": "test", "segmentation": [[0, 0, 1, 0, 1, 1, 0, 1]]},
            {
                "task": "semantic_segmentation",
                "image_id": "test",
                "category_id": "test",
                "bbox": [0, 0, 1, 1],
                "segmentation": [[0, 0, 1, 0, 1, 1, 0, 1]],
                "area": 1,
                "iscrowd": 0,
                "is_prediction": False,
            },
        ),
        (
            TaskType.SEMANTIC_SEGMENTATION,
            {"category_id": "test", "segmentation": [[0, 0, 1, 0, 1, 1, 0, 1]]},
            TypeError,
        ),
        (
            TaskType.SEMANTIC_SEGMENTATION,
            {"image_id": None, "category_id": "test", "segmentation": [[0, 0, 1, 0, 1, 1, 0, 1]]},
            ValueError,
        ),
        (
            TaskType.SEMANTIC_SEGMENTATION,
            {"image_id": "test", "segmentation": [[0, 0, 1, 0, 1, 1, 0, 1]]},
            TypeError,
        ),
        (
            TaskType.SEMANTIC_SEGMENTATION,
            {"image_id": "test", "category_id": "test", "segmentation": [[0, 0, 1, 0, 1, 1, 0]]},
            ValueError,
        ),
        (
            TaskType.SEMANTIC_SEGMENTATION,
            {
                "image_id": "test",
                "category_id": "test",
                "segmentation": [[0, 0, 1, 0, 1, 1, 0, 1, 1]],
            },
            ValueError,
        ),
        (
            TaskType.SEMANTIC_SEGMENTATION,
            {
                "image_id": "test",
                "category_id": "test",
                "segmentation": [[0, 0, 1, 0, 1, 1, 0, 1]],
                "area": "asdf",
            },
            ValueError,
        ),
    ],
)
def test_semantic_segmentation_annotation_info(task, kwargs, expected_output):
    _test_annotation_info(task, kwargs, expected_output)


@pytest.mark.parametrize(
    "task, kwargs, expected_output",
    [
        (
            TaskType.INSTANCE_SEGMENTATION,
            {"image_id": "test", "category_id": "test", "segmentation": [[0, 0, 1, 0, 1, 1, 0, 1]]},
            {
                "task": "instance_segmentation",
                "image_id": "test",
                "category_id": "test",
                "bbox": [0, 0, 1, 1],
                "segmentation": [[0, 0, 1, 0, 1, 1, 0, 1]],
                "area": 1,
                "iscrowd": 0,
                "is_prediction": False,
            },
        ),
        (
            TaskType.INSTANCE_SEGMENTATION,
            {
                "image_id": "test",
                "category_id": "test",
                "segmentation": [[0, 0, 1, 0, 1, 1, 0, 1]],
                "score": 0.5,
            },
            {
                "task": "instance_segmentation",
                "image_id": "test",
                "category_id": "test",
                "bbox": [0, 0, 1, 1],
                "segmentation": [[0, 0, 1, 0, 1, 1, 0, 1]],
                "area": 1,
                "iscrowd": 0,
                "score": 0.5,
                "is_prediction": True,
            },
        ),
        (
            TaskType.INSTANCE_SEGMENTATION,
            {"category_id": "test", "segmentation": [[0, 0, 1, 0, 1, 1, 0, 1]]},
            TypeError,
        ),
        (
            TaskType.INSTANCE_SEGMENTATION,
            {"image_id": None, "category_id": "test", "segmentation": [[0, 0, 1, 0, 1, 1, 0, 1]]},
            ValueError,
        ),
        (
            TaskType.INSTANCE_SEGMENTATION,
            {"image_id": "test", "segmentation": [[0, 0, 1, 0, 1, 1, 0, 1]]},
            TypeError,
        ),
        (
            TaskType.INSTANCE_SEGMENTATION,
            {"image_id": "test", "category_id": "test", "segmentation": [[0, 0, 1, 0, 1, 1, 0]]},
            ValueError,
        ),
        (
            TaskType.INSTANCE_SEGMENTATION,
            {
                "image_id": "test",
                "category_id": "test",
                "segmentation": [[0, 0, 1, 0, 1, 1, 0, 1, 1]],
            },
            ValueError,
        ),
        (
            TaskType.INSTANCE_SEGMENTATION,
            {
                "image_id": "test",
                "category_id": "test",
                "segmentation": [[0, 0, 1, 0, 1, 1, 0, 1]],
                "area": "asdf",
            },
            ValueError,
        ),
    ],
)
def test_instance_segmentation_annotation_info(task, kwargs, expected_output):
    _test_annotation_info(task, kwargs, expected_output)
