import pytest

from waffle_dough.type import DataType, TaskType


def _type_check(type_member, test_value):
    assert type_member == test_value
    assert type_member == test_value.upper()
    assert type_member == test_value.lower()
    assert type_member.lower() == test_value.lower()
    assert type_member.upper() != test_value.lower()
    assert type_member.upper() == test_value.upper()
    assert type_member.lower() != test_value.upper()

    assert type_member != "something not member"


def _member_check(type_class, test_value):
    assert test_value.lower() in list(type_class)
    assert test_value.upper() in list(type_class)


@pytest.mark.parametrize("data_type, test_value", [[DataType.COCO, "coco"], [DataType.YOLO, "yolo"]])
def test_data_type(data_type, test_value):
    _type_check(data_type, test_value)
    _member_check(DataType, test_value)


@pytest.mark.parametrize(
    "task_type, test_value",
    [
        [TaskType.CLASSIFICATION, "classification"],
        [TaskType.OBJECT_DETECTION, "object_detection"],
        [TaskType.SEMANTIC_SEGMENTATION, "semantic_segmentation"],
        [TaskType.INSTANCE_SEGMENTATION, "instance_segmentation"],
        [TaskType.KEYPOINT_DETECTION, "keypoint_detection"],
        [TaskType.TEXT_RECOGNITION, "text_recognition"],
        [TaskType.REGRESSION, "regression"],
    ],
)
def test_task_type(task_type, test_value):
    _type_check(task_type, test_value)
    _member_check(TaskType, test_value)
