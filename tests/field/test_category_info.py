import pytest

from waffle_dough.field import CategoryInfo
from waffle_dough.type import TaskType


def _test_category_info(task, kwargs, expected_output):
    new_function = getattr(CategoryInfo, task.lower())
    if not isinstance(expected_output, dict):
        with pytest.raises(expected_output):
            new_function(**kwargs)
    else:
        category_1 = new_function(**kwargs)
        output = category_1.to_dict()
        output.pop("id")
        assert output == expected_output

        category_2 = CategoryInfo.from_dict(task=task, d=kwargs)
        assert category_1 == category_2

        kwargs.update({"name": kwargs["name"] + "2"})
        category_3 = new_function(**kwargs)
        assert category_1 != category_3


@pytest.mark.parametrize(
    "task, kwargs, expected_output",
    [
        (
            TaskType.CLASSIFICATION,
            {"name": "test", "supercategory": "test"},
            {
                "task": "classification",
                "name": "test",
                "supercategory": "test",
            },
        ),
        (
            TaskType.CLASSIFICATION,
            {"supercategory": "test"},
            TypeError,
        ),
        (
            TaskType.CLASSIFICATION,
            {"name": None, "supercategory": "test"},
            ValueError,
        ),
    ],
)
def test_classification_category_info(task, kwargs, expected_output):
    _test_category_info(task, kwargs, expected_output)


@pytest.mark.parametrize(
    "task, kwargs, expected_output",
    [
        (
            TaskType.OBJECT_DETECTION,
            {"name": "test", "supercategory": "test"},
            {
                "task": "object_detection",
                "name": "test",
                "supercategory": "test",
            },
        ),
        (
            TaskType.OBJECT_DETECTION,
            {"supercategory": "test"},
            TypeError,
        ),
        (
            TaskType.OBJECT_DETECTION,
            {"name": None, "supercategory": "test"},
            ValueError,
        ),
    ],
)
def test_object_detection_category_info(task, kwargs, expected_output):
    _test_category_info(task, kwargs, expected_output)


@pytest.mark.parametrize(
    "task, kwargs, expected_output",
    [
        (
            TaskType.SEMANTIC_SEGMENTATION,
            {"name": "test", "supercategory": "test"},
            {
                "task": "semantic_segmentation",
                "name": "test",
                "supercategory": "test",
            },
        ),
        (
            TaskType.SEMANTIC_SEGMENTATION,
            {"supercategory": "test"},
            TypeError,
        ),
        (
            TaskType.SEMANTIC_SEGMENTATION,
            {"name": None, "supercategory": "test"},
            ValueError,
        ),
    ],
)
def test_semantic_segmentation_category_info(task, kwargs, expected_output):
    _test_category_info(task, kwargs, expected_output)


@pytest.mark.parametrize(
    "task, kwargs, expected_output",
    [
        (
            TaskType.INSTANCE_SEGMENTATION,
            {"name": "test", "supercategory": "test"},
            {
                "task": "instance_segmentation",
                "name": "test",
                "supercategory": "test",
            },
        ),
        (
            TaskType.INSTANCE_SEGMENTATION,
            {"supercategory": "test"},
            TypeError,
        ),
        (
            TaskType.INSTANCE_SEGMENTATION,
            {"name": None, "supercategory": "test"},
            ValueError,
        ),
    ],
)
def test_instance_segmentation_category_info(task, kwargs, expected_output):
    _test_category_info(task, kwargs, expected_output)


@pytest.mark.parametrize(
    "task, kwargs, expected_output",
    [
        (
            TaskType.KEYPOINT_DETECTION,
            {
                "name": "test",
                "supercategory": "test",
                "keypoints": ["test1", "test2"],
                "skeleton": [[0, 1]],
            },
            {
                "task": "keypoint_detection",
                "name": "test",
                "supercategory": "test",
                "keypoints": ["test1", "test2"],
                "skeleton": [[0, 1]],
            },
        ),
        (
            TaskType.KEYPOINT_DETECTION,
            {
                "name": "test",
                "supercategory": "test",
                "keypoints": ["test1", "test2"],
                "skeleton": [[0, 1, 2]],
            },
            ValueError,
        ),
        (
            TaskType.KEYPOINT_DETECTION,
            {
                "name": "test",
                "supercategory": "test",
                "keypoints": ["test1", "test2"],
                "skeleton": [[0, 1], [1, 2]],
            },
            ValueError,
        ),
        (
            TaskType.KEYPOINT_DETECTION,
            {"supercategory": "test", "keypoints": ["test1", "test2"], "skeleton": [[0, 1]]},
            TypeError,
        ),
        (
            TaskType.KEYPOINT_DETECTION,
            {
                "name": None,
                "supercategory": "test",
                "keypoints": ["test1", "test2"],
                "skeleton": [[0, 1]],
            },
            ValueError,
        ),
    ],
)
def test_keypoint_detection_category_info(task, kwargs, expected_output):
    _test_category_info(task, kwargs, expected_output)


@pytest.mark.parametrize(
    "task, kwargs, expected_output",
    [
        (
            TaskType.TEXT_RECOGNITION,
            {"name": "test", "supercategory": "test"},
            {
                "task": "text_recognition",
                "name": "test",
                "supercategory": "test",
            },
        ),
        (
            TaskType.TEXT_RECOGNITION,
            {"supercategory": "test"},
            TypeError,
        ),
        (
            TaskType.TEXT_RECOGNITION,
            {"name": None, "supercategory": "test"},
            ValueError,
        ),
    ],
)
def test_text_recognition_category_info(task, kwargs, expected_output):
    _test_category_info(task, kwargs, expected_output)


@pytest.mark.parametrize(
    "task, kwargs, expected_output",
    [
        (
            TaskType.REGRESSION,
            {"name": "test", "supercategory": "test"},
            {
                "task": "regression",
                "name": "test",
                "supercategory": "test",
            },
        ),
        (
            TaskType.REGRESSION,
            {"supercategory": "test"},
            TypeError,
        ),
        (
            TaskType.REGRESSION,
            {"name": None, "supercategory": "test"},
            ValueError,
        ),
    ],
)
def test_regression_category_info(task, kwargs, expected_output):
    _test_category_info(task, kwargs, expected_output)
