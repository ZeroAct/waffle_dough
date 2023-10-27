import pytest

from waffle_dough.field import ImageInfo
from waffle_dough.type import TaskType


def _test_image_info(task, kwargs, expected_output):
    new_function = getattr(ImageInfo, task.lower())
    if not isinstance(expected_output, dict):
        with pytest.raises(expected_output):
            new_function(**kwargs)
    else:
        image_1 = new_function(**kwargs)
        output = image_1.to_dict()
        output.pop("id")
        assert output == expected_output

        image_2 = ImageInfo.from_dict(task=task, d=kwargs)
        assert image_1 == image_2

        kwargs.update({"file_name": kwargs["file_name"] + "2"})
        kwargs.pop("task")
        image_3 = new_function(**kwargs)
        assert image_1 != image_3

        kwargs.update({"original_file_name": image_1.file_name})
        image_3 = new_function(**kwargs)
        assert image_1 != image_3

        kwargs.update({"width": image_1.width + 1})
        image_3 = new_function(**kwargs)
        assert image_1 != image_3

        kwargs.update({"height": image_1.height + 1})
        image_3 = new_function(**kwargs)
        assert image_1 != image_3


@pytest.mark.parametrize(
    "task, kwargs, expected_output",
    [
        (
            TaskType.AGNOSTIC,
            {"file_name": "test.jpg", "width": 100, "height": 100},
            {
                "task": "agnostic",
                "file_name": "test.jpg",
                "width": 100,
                "height": 100,
                "labeled": False,
                "split": "unset",
            },
        ),
        (
            TaskType.AGNOSTIC,
            {"file_name": "test.jpg", "width": 100, "height": 100, "original_file_name": "test.jpg"},
            {
                "task": "agnostic",
                "file_name": "test.jpg",
                "width": 100,
                "height": 100,
                "original_file_name": "test.jpg",
                "labeled": False,
                "split": "unset",
            },
        ),
        (
            TaskType.AGNOSTIC,
            {
                "file_name": "test.jpg",
                "width": 100,
                "height": 100,
                "original_file_name": "test.jpg",
                "date_captured": "2021-01-01",
            },
            {
                "task": "agnostic",
                "file_name": "test.jpg",
                "width": 100,
                "height": 100,
                "original_file_name": "test.jpg",
                "date_captured": "2021-01-01",
                "labeled": False,
                "split": "unset",
            },
        ),
        (
            TaskType.AGNOSTIC,
            {"width": 100, "height": 100},
            TypeError,
        ),
        (
            TaskType.AGNOSTIC,
            {"file_name": None, "width": 100, "height": 100},
            ValueError,
        ),
        (
            TaskType.AGNOSTIC,
            {"file_name": "test.jpg", "width": 0, "height": 100},
            ValueError,
        ),
        (
            TaskType.AGNOSTIC,
            {"file_name": "test.jpg", "width": 100, "height": 0},
            ValueError,
        ),
    ],
)
def test_agnostic_image_info(task, kwargs, expected_output):
    _test_image_info(task, kwargs, expected_output)
