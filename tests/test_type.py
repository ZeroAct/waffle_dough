from waffle_dough.type.base_type import BaseType
from waffle_dough.type.data_type import DataType
from waffle_dough.type.task_type import TaskType


def test_base_type():

    class TestType(BaseType):
        TEST = "test"
        TEST2 = "test2"

    assert TestType.TEST == "test"
    assert TestType.TEST == "TEST"
    assert TestType.TEST != "Test2"

    assert TestType.TEST2 == "test2"
    assert TestType.TEST2 == "TEST2"
    assert TestType.TEST2 != "Test"

    assert TestType.TEST == TestType.TEST
    assert TestType.TEST != TestType.TEST2

    assert TestType.TEST.upper() == "TEST"
    assert TestType.TEST2.upper() == "TEST2"

    assert TestType.TEST.lower() == "test"
    assert TestType.TEST2.lower() == "test2"

    assert "test" in [TestType.TEST, TestType.TEST2]
    assert "test" in list(TestType)

    test_dict = {
        TestType.TEST: "test",
        TestType.TEST2: "test2"
    }

    assert test_dict[TestType.TEST] == "test"

    assert TestType.TEST in test_dict

    assert test_dict["test"] == "test"


def test_data_type():
    assert DataType.COCO == "coco"
    assert DataType.COCO == "COCO"
    assert DataType.COCO != "YOLO"

    assert DataType.YOLO == "yolo"
    assert DataType.YOLO == "YOLO"
    assert DataType.YOLO != "COCO"

    assert DataType.COCO == DataType.COCO
    assert DataType.COCO != DataType.YOLO

    assert DataType.COCO.upper() == "COCO"
    assert DataType.YOLO.upper() == "YOLO"

    assert DataType.COCO.lower() == "coco"
    assert DataType.YOLO.lower() == "yolo"

    assert "coco" in [DataType.COCO, DataType.YOLO]
    assert "coco" in list(DataType)

    data_dict = {
        DataType.COCO: "coco",
        DataType.YOLO: "yolo"
    }

    assert data_dict[DataType.COCO] == "coco"

    assert DataType.COCO in data_dict

    assert data_dict["coco"] == "coco"


def test_task_type():
    assert TaskType.CLASSIFICATION == "classification"
    assert TaskType.CLASSIFICATION == "CLASSIFICATION"
    assert TaskType.CLASSIFICATION != "OBJECT_DETECTION"

    assert TaskType.OBJECT_DETECTION == "object_detection"
    assert TaskType.OBJECT_DETECTION == "OBJECT_DETECTION"
    assert TaskType.OBJECT_DETECTION != "CLASSIFICATION"

    assert TaskType.INSTANCE_SEGMENTATION == "instance_segmentation"
    assert TaskType.INSTANCE_SEGMENTATION == "INSTANCE_SEGMENTATION"
    assert TaskType.INSTANCE_SEGMENTATION != "CLASSIFICATION"

    assert TaskType.CLASSIFICATION == TaskType.CLASSIFICATION
    assert TaskType.CLASSIFICATION != TaskType.OBJECT_DETECTION

    assert TaskType.OBJECT_DETECTION == TaskType.OBJECT_DETECTION
    assert TaskType.OBJECT_DETECTION != TaskType.CLASSIFICATION

    assert TaskType.INSTANCE_SEGMENTATION == TaskType.INSTANCE_SEGMENTATION
    assert TaskType.INSTANCE_SEGMENTATION != TaskType.CLASSIFICATION

    assert TaskType.CLASSIFICATION.upper() == "CLASSIFICATION"
    assert TaskType.OBJECT_DETECTION.upper() == "OBJECT_DETECTION"
    assert TaskType.INSTANCE_SEGMENTATION.upper() == "INSTANCE_SEGMENTATION"

    assert TaskType.CLASSIFICATION.lower() == "classification"
    assert TaskType.OBJECT_DETECTION.lower() == "object_detection"
    assert TaskType.INSTANCE_SEGMENTATION.lower() == "instance_segmentation"

    assert "classification" in [TaskType.CLASSIFICATION, TaskType.OBJECT_DETECTION, TaskType.INSTANCE_SEGMENTATION]
    assert "classification" in list(TaskType)

    task_dict = {
        TaskType.CLASSIFICATION: "classification",
        TaskType.OBJECT_DETECTION: "object_detection",
        TaskType.INSTANCE_SEGMENTATION: "instance_segmentation"
    }

    assert task_dict[TaskType.CLASSIFICATION] == "classification"

    assert TaskType.CLASSIFICATION in task_dict

    assert task_dict["classification"] == "classification"
