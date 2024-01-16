import logging
from pathlib import Path
from typing import Union

import natsort
from waffle_utils.file import io, search

from waffle_dough.exception import *
from waffle_dough.field import AnnotationInfo, CategoryInfo, ImageInfo
from waffle_dough.image import io as image_io
from waffle_dough.type import SegmentationType, SplitType, TaskType

from .base_adapter import BaseAdapter

logger = logging.getLogger(__name__)


SPLIT_MAP = {
    "train": SplitType.TRAIN,
    "val": SplitType.VALIDATION,
    "test": SplitType.TEST,
}
INV_SPLIT_MAP = {v: k for k, v in SPLIT_MAP.items()}


class YOLOAdapter(BaseAdapter):
    def __init__(
        self,
        image_dict: dict[str, ImageInfo] = None,
        annotation_dict: dict[str, AnnotationInfo] = None,
        category_dict: dict[str, CategoryInfo] = None,
        task: Union[str, TaskType] = TaskType.OBJECT_DETECTION,
        callbacks: list[callable] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            image_dict=image_dict,
            annotation_dict=annotation_dict,
            category_dict=category_dict,
            task=task,
            callbacks=[] + (callbacks if callbacks else []),
            *args,
            **kwargs,
        )

        if self.task not in [
            TaskType.OBJECT_DETECTION,
            TaskType.INSTANCE_SEGMENTATION,
            TaskType.CLASSIFICATION,
            TaskType.KEYPOINT_DETECTION,  # TODO: Add support for keypoint detection
        ]:
            raise DatasetAdapterTaskError(f"Task {self.task} is not supported by YOLO format.")
        pass

    def _load_data_yaml_file(self, directory: Union[str, Path]) -> dict:
        directory = Path(directory)

        files = search.get_files(directory, recursive=False, extension=[".yaml", ".yml"])
        if len(files) == 1:
            return io.load_yaml(files[0])
        else:
            raise DatasetAdapterImportError(
                f"Cannot suggest a yaml file to use. Found {files} in {directory}."
            )

    def _import_object_detection(
        self,
        yolo_root_dir: Union[str, Path],
    ):
        data = self._load_data_yaml_file(yolo_root_dir)

        names = data.get("names", None)
        if names is None:
            raise DatasetAdapterImportError(f'Cannot find the "names" field in yaml file.\n{data}')

        self.category_dict = {}
        yolo_id_to_category_id = {}
        for i, name in names.items():
            category = CategoryInfo.object_detection(
                name=name,
            )
            self.category_dict[category.id] = category
            yolo_id_to_category_id[i] = category.id

        image_files = search.get_image_files(yolo_root_dir, recursive=True)

        self.run_callback_hooks("on_loop_start", len(image_files))

        self.image_dict = {}
        self.annotation_dict = {}
        for image_file in image_files:
            image_file = Path(image_file)

            rel_path = image_file.relative_to(yolo_root_dir)
            parts = rel_path.parts
            split = parts[0]

            np_image = image_io.cv2_imread(image_file)
            image = ImageInfo.agnostic(
                width=np_image.shape[1],
                height=np_image.shape[0],
                original_file_name=rel_path,
                split=SPLIT_MAP.get(split, SplitType.UNSET),
            )
            self.image_dict[image.id] = image

            annotation_file = Path(rel_path).with_suffix(".txt")
            annotation_file = Path(yolo_root_dir, split, "labels", *annotation_file.parts[2:])

            if not annotation_file.exists():
                logger.warning(f"Annotation file {annotation_file} does not exist.")

            if annotation_file.exists():
                with open(str(annotation_file)) as f:
                    lines = f.readlines()
                    for line in lines:
                        category_id, cx, cy, w, h = map(float, line.split())
                        x1 = cx - w / 2
                        y1 = cy - h / 2

                        annotation = AnnotationInfo.object_detection(
                            image_id=image.id,
                            category_id=yolo_id_to_category_id[int(category_id)],
                            bbox=[x1, y1, w, h],
                        )
                        self.annotation_dict[annotation.id] = annotation

            self.run_callback_hooks("on_step_end")

        self.run_callback_hooks("on_loop_end")

    def _import_instance_segmentation(
        self,
        yolo_root_dir: Union[str, Path],
    ):
        data = self._load_data_yaml_file(yolo_root_dir)

        names = data.get("names", None)
        if names is None:
            raise DatasetAdapterImportError(f'Cannot find the "names" field in yaml file.\n{data}')

        self.category_dict = {}
        yolo_id_to_category_id = {}
        for i, name in names.items():
            category = CategoryInfo.instance_segmentation(
                name=name,
            )
            self.category_dict[category.id] = category
            yolo_id_to_category_id[i] = category.id

        image_files = search.get_image_files(yolo_root_dir, recursive=True)

        self.run_callback_hooks("on_loop_start", len(image_files))

        self.image_dict = {}
        self.annotation_dict = {}
        for image_file in image_files:
            image_file = Path(image_file)

            rel_path = image_file.relative_to(yolo_root_dir)
            parts = rel_path.parts
            split = parts[0]

            np_image = image_io.cv2_imread(image_file)
            image = ImageInfo.agnostic(
                width=np_image.shape[1],
                height=np_image.shape[0],
                original_file_name=rel_path,
                split=SPLIT_MAP.get(split, SplitType.UNSET),
            )
            self.image_dict[image.id] = image

            annotation_file = Path(rel_path).with_suffix(".txt")
            annotation_file = Path(yolo_root_dir, split, "labels", *annotation_file.parts[2:])

            if not annotation_file.exists():
                logger.warning(f"Annotation file {annotation_file} does not exist.")

            if annotation_file.exists():
                with open(str(annotation_file)) as f:
                    lines = f.readlines()
                    for line in lines:
                        category_id, *pts = map(float, line.split())

                        annotation = AnnotationInfo.instance_segmentation(
                            image_id=image.id,
                            category_id=yolo_id_to_category_id[int(category_id)],
                            segmentation=[
                                pts
                            ],  # TODO: yolo format does not support multiple segmentations
                        )

                        self.annotation_dict[annotation.id] = annotation

            self.run_callback_hooks("on_step_end")

        self.run_callback_hooks("on_loop_end")

    def _import_classiffication(
        self,
        yolo_root_dir: Union[str, Path],
    ):
        image_files = search.get_image_files(yolo_root_dir, recursive=True)

        self.run_callback_hooks("on_loop_start", len(image_files))

        self.category_dict = {}
        self.image_dict = {}
        self.annotation_dict = {}
        category_name_to_id = {}

        for image_file in image_files:
            image_file = Path(image_file)

            rel_path = image_file.relative_to(yolo_root_dir)
            parts = rel_path.parts
            split = parts[0]
            category_name = parts[1]

            if category_name not in category_name_to_id:
                category = CategoryInfo.classification(
                    name=category_name,
                )
                category_name_to_id[category_name] = category.id
                self.category_dict[category.id] = category

            np_image = image_io.cv2_imread(image_file)
            image = ImageInfo.agnostic(
                width=np_image.shape[1],
                height=np_image.shape[0],
                original_file_name=rel_path,
                split=SPLIT_MAP.get(split, SplitType.UNSET),
            )
            self.image_dict[image.id] = image

            annotation = AnnotationInfo.classification(
                image_id=image.id,
                category_id=category_name_to_id[category_name],
            )
            self.annotation_dict[annotation.id] = annotation

            self.run_callback_hooks("on_step_end")

        self.run_callback_hooks("on_loop_end")

    def import_target(
        self,
        yolo_root_dir: Union[str, Path],
    ) -> "YOLOAdapter":
        if self.task == TaskType.OBJECT_DETECTION:
            self._import_object_detection(yolo_root_dir)
        elif self.task == TaskType.INSTANCE_SEGMENTATION:
            self._import_instance_segmentation(yolo_root_dir)
        elif self.task == TaskType.CLASSIFICATION:
            self._import_classiffication(yolo_root_dir)
        else:
            raise DatasetAdapterTaskError(f"Task {self.task} is not supported by YOLO format.")

        return self

    def _export_object_detection(
        self,
        yolo_root_dir: Union[str, Path],
        image_path_getter: callable,
    ):
        yolo_root_dir = Path(yolo_root_dir)
        io.make_directory(yolo_root_dir)

        data = {
            "path": str(yolo_root_dir),
            "names": {},
        }
        name_to_yolo_id = {}
        for i, category in enumerate(self.category_dict.values()):
            data["names"][i] = category.name
            name_to_yolo_id[category.name] = i

        self.run_callback_hooks("on_loop_start", len(self.annotation_dict))

        image_check = {}
        for annotation in self.annotation_dict.values():
            image = self.image_dict[annotation.image_id]

            split = INV_SPLIT_MAP.get(image.split, None)
            if split is None:
                self.run_callback_hooks("on_step_end")
                continue

            if split not in data:
                data[split] = split

            image_path = image_path_getter(image)
            if image_path not in image_check:
                io.copy_file(
                    image_path,
                    Path(yolo_root_dir, split, "images", f"{image.id}{image.ext}"),
                    create_directory=True,
                )
                image_check[image_path] = True

            annotation_file = Path(yolo_root_dir, split, "labels", f"{image.id}.txt")
            annotation_file.parent.mkdir(parents=True, exist_ok=True)

            x1, y1, w, h = annotation.bbox
            cx = x1 + w / 2
            cy = y1 + h / 2
            with open(str(annotation_file), "a") as f:
                f.write(
                    f"{name_to_yolo_id[self.category_dict[annotation.category_id].name]} {cx} {cy} {w} {h}\n"
                )

            self.run_callback_hooks("on_step_end")

        io.save_yaml(data, Path(yolo_root_dir, "data.yaml"))

        self.run_callback_hooks("on_loop_end")

    def _export_instance_segmentation(
        self,
        yolo_root_dir: Union[str, Path],
        image_path_getter: callable,
    ):
        yolo_root_dir = Path(yolo_root_dir)
        io.make_directory(yolo_root_dir)

        data = {
            "path": str(yolo_root_dir),
            "names": {},
        }
        name_to_yolo_id = {}
        for i, category in enumerate(self.category_dict.values()):
            data["names"][i] = category.name
            name_to_yolo_id[category.name] = i

        self.run_callback_hooks("on_loop_start", len(self.annotation_dict))

        image_check = {}
        for annotation in self.annotation_dict.values():
            image = self.image_dict[annotation.image_id]

            split = INV_SPLIT_MAP.get(image.split, None)
            if split is None:
                self.run_callback_hooks("on_step_end")
                continue

            if split not in data:
                data[split] = split

            image_path = image_path_getter(image)
            if image_path not in image_check:
                io.copy_file(
                    image_path,
                    Path(yolo_root_dir, split, "images", f"{image.id}{image.ext}"),
                    create_directory=True,
                )
                image_check[image_path] = True

            annotation_file = Path(yolo_root_dir, split, "labels", f"{image.id}.txt")
            annotation_file.parent.mkdir(parents=True, exist_ok=True)

            if len(annotation.segmentation) > 1:
                logger.warning(
                    f"YOLO format does not support multiple segmentations. Using the first segmentation."
                )
            segmentation = annotation.segmentation[0]
            with open(str(annotation_file), "a") as f:
                f.write(
                    f"{name_to_yolo_id[self.category_dict[annotation.category_id].name]} {' '.join(map(str, segmentation))}\n"
                )

            self.run_callback_hooks("on_step_end")

        io.save_yaml(data, Path(yolo_root_dir, "data.yaml"))

        self.run_callback_hooks("on_loop_end")

    def _export_classification(
        self,
        yolo_root_dir: Union[str, Path],
        image_path_getter: callable,
    ):
        yolo_root_dir = Path(yolo_root_dir)
        io.make_directory(yolo_root_dir)

        self.run_callback_hooks("on_loop_start", len(self.annotation_dict))

        image_check = {}
        for annotation in self.annotation_dict.values():
            image = self.image_dict[annotation.image_id]

            split = INV_SPLIT_MAP.get(image.split, None)
            if split is None:
                self.run_callback_hooks("on_step_end")
                continue

            image_path = image_path_getter(image)
            if image_path in image_check:
                raise DatasetAdapterExportError(
                    f"Image {image_path} is duplicated. Only one annotation per image is allowed."
                )
            image_check[image_path] = True

            io.copy_file(
                image_path,
                Path(
                    yolo_root_dir,
                    split,
                    f"{self.category_dict[annotation.category_id].name}",
                    f"{image.id}{image.ext}",
                ),
                create_directory=True,
            )

            self.run_callback_hooks("on_step_end")

        self.run_callback_hooks("on_loop_end")

    def export_target(
        self,
        yolo_root_dir: Union[str, Path],
        image_path_getter: callable,
    ) -> "YOLOAdapter":
        if self.task == TaskType.OBJECT_DETECTION:
            self._export_object_detection(yolo_root_dir, image_path_getter)
        elif self.task == TaskType.INSTANCE_SEGMENTATION:
            self._export_instance_segmentation(yolo_root_dir, image_path_getter)
        elif self.task == TaskType.CLASSIFICATION:
            self._export_classification(yolo_root_dir, image_path_getter)
        else:
            raise DatasetAdapterTaskError(f"Task {self.task} is not supported by YOLO format.")

        return self
